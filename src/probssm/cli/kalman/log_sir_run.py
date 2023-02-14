import argparse
import functools
import logging
import pathlib
import time
from datetime import datetime

import numpy as np
import probnum as pn
import scipy.linalg
import scipy.special
from probnum import filtsmooth, problems, randprocs, randvars
from scipy.optimize import minimize_scalar
from scipy import stats

import probssm

from ._likelihoods import LogSIRLikelihood

logging.basicConfig(
    level=logging.INFO,
    format=">>> %(message)s",
)


def parse_args():
    parser = argparse.ArgumentParser(fromfile_prefix_chars="@")

    parser.add_argument("--logdir", type=str, default="./run")
    parser.add_argument("--logdir-suffix", type=str, required=False)

    data_arg_group = parser.add_argument_group(title="Data")
    data_arg_group.add_argument("--country", type=str, default="Germany")
    data_arg_group.add_argument(
        "--scaling", type=str, choices=["none", "cpt"], default="cpt"
    )
    data_arg_group.add_argument("--num-train", type=int, required=False)
    data_arg_group.add_argument("--num-extrapolate", type=int, default=0)
    data_arg_group.add_argument("--skip-first", type=int, default=0)
    data_arg_group.add_argument("--num-data-points", type=int, default=556)

    model_arg_group = parser.add_argument_group(title="Model Hyperparameters")
    model_arg_group.add_argument("--sigmoid-slope", type=float, default=0.01)
    _default_filter_stepsize = 1.0 / 24.0
    model_arg_group.add_argument("--filter-step-size", type=float, default=_default_filter_stepsize)
    model_arg_group.add_argument("--purely-mechanistic", action="store_true")
    model_arg_group.add_argument("--purely-data", action="store_true")
    model_arg_group.add_argument("--filter-smoothing", type=int, default=1)

    parser.add_argument("--num-samples", type=int, default=0)
    parser.add_argument("--save-intermediate-filtering", type=int, required=False)
    parser.add_argument("--pn-forward-implementation", type=str, default="sqrt")
    parser.add_argument("--pn-backward-implementation", type=str, default="sqrt")

    x_process_arg_group = parser.add_argument_group(title="X-process")
    x_process_arg_group.add_argument("--x-process-ordint", type=int, default=2)

    beta_process_arg_group = parser.add_argument_group(title="beta-process")

    beta_process_arg_group.add_argument("--beta-process-ordint", type=int, default=0)

    # Checks
    arg_namespace = parser.parse_args()
    if arg_namespace.purely_mechanistic and arg_namespace.purely_data:
        raise ValueError("Can only set --purely-mechanistic XOR --purely-data.")

    return arg_namespace


def main():
    args = parse_args()

    logging.info("===== Starting Log SIR data experiment =====")

    rng = np.random.default_rng(seed=123)

    STATE_DIM = 3 #S, I, R
    OBSERVATION_DIM = 1 #We observe Cumulative cases, i.e. C(t) = S0 - S(t)

    X_PROCESS_NUM = 0 #Index of state process
    BETA_PROCESS_NUM = 1 #Index for beta process

    # Set up log dir
    if args.logdir_suffix is None:
        logdir_suffix = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    else:
        logdir_suffix = args.logdir_suffix
    log_dir = pathlib.Path(f"{args.logdir}_{logdir_suffix}").absolute()
    log_dir.mkdir(parents=True, exist_ok=False)

    logging.info(f"Logging to {log_dir}")

    # ##################################################################################
    # DATA
    # ##################################################################################

    # COVID-data
    day_zero, date_range_x, SIR_data, population = probssm.data.load_COVID_data(
        country=args.country, num_data_points=args.num_data_points, include_death=False, skip_first=args.skip_first,
    )

    num_covid_data_points = SIR_data.shape[0]

    if args.scaling == "cpt":
        cases_per_thousand_scaling = 1e3 / population

        population = population * cases_per_thousand_scaling
        assert np.isclose(population, 1000.0), population

        SIR_data = SIR_data * cases_per_thousand_scaling

    # Transform data to log
    SIR_data += 1e-5
    SIR_data = np.log(SIR_data)

    # Pseudo observations for ODE measurements
    zero_data = np.zeros(STATE_DIM, dtype=np.float64)

    # Split into train and validation set
    if args.num_train is None:
        num_train = num_covid_data_points
    elif args.num_train <= 0:
        num_train = num_covid_data_points + args.num_train
    else:
        num_train = args.num_train

    test_set_conditions = [
        lambda i: i >= num_train,
        # lambda i: i % 2 != 0,
    ]

    train_idcs, val_idcs = probssm.split_train_test(
        all_idcs=np.arange(num_covid_data_points, dtype=int),
        list_lambda_predicates_test=test_set_conditions,
    )



    def get_states(beta_process_lengthscale = 75, beta_process_diffusion = 0.05, x_process_diffusion = 0.05, ode_measurement_cov = 5e-7, data_measurement_cov = 1e-9, gamma = 0.06, beta_prior_mean = 0.1, sir_0 = np.array(SIR_data[0, :3]), init_sir_vel = 1e-3, init_beta_vel = 0.0, save=False):
        
        forward_implementation = args.pn_forward_implementation
        backward_implementation = args.pn_backward_implementation

        #Specify 
        ode_transition = pn.randprocs.markov.integrator.IntegratedWienerTransition(
            num_derivatives=args.x_process_ordint,
            wiener_process_dimension=STATE_DIM,
            forward_implementation=forward_implementation,
            backward_implementation=backward_implementation,
        )

        ode_transition._dispersion_matrix = (
            ode_transition._dispersion_matrix * x_process_diffusion
        )

        lf_transition = pn.randprocs.markov.integrator.MaternTransition(
            num_derivatives=args.beta_process_ordint,
            wiener_process_dimension=1,
            lengthscale=beta_process_lengthscale,
            forward_implementation=forward_implementation,
            backward_implementation=backward_implementation,
        )

        lf_transition._dispersion_matrix = (
            lf_transition._dispersion_matrix * beta_process_diffusion
        )

        prior_transition = probssm.stacked_ssm.StackedTransition(
            transitions=(ode_transition, lf_transition),
            forward_implementation=forward_implementation,
            backward_implementation=backward_implementation,
        )

        # Set up initial conditions

        # ##################################################################################
        # ODE LIKELIHOOD
        # ##################################################################################

        # Link functions

        sigmoid_x_offset = -scipy.special.logit(beta_prior_mean)
        beta_link_fn = functools.partial(
            probssm.util.sloped_sigmoid,
            slope=args.sigmoid_slope,
            x_offset=sigmoid_x_offset,
        )
        beta_link_fn_deriv = functools.partial(
            probssm.util.d_sloped_sigmoid,
            slope=args.sigmoid_slope,
            x_offset=sigmoid_x_offset,
        )
        beta_inverse_link_fn = (
            lambda x: (scipy.special.logit(x) + sigmoid_x_offset) / args.sigmoid_slope
        )

        assert np.isclose(beta_link_fn(beta_inverse_link_fn(beta_prior_mean)), beta_prior_mean)
        assert np.isclose(beta_link_fn(0.0), beta_prior_mean)

        ode_parameters = {"gamma": gamma,"population_count": population}

        ode_likelihood = LogSIRLikelihood(
            prior=prior_transition,
            ode_parameters=ode_parameters,
            beta_link_fn=beta_link_fn,
            beta_link_fn_deriv=beta_link_fn_deriv,
        )

        process_idcs = prior_transition.state_idcs

        # Mean

        logging.debug(f"Initial SIR mean: {np.exp(sir_0)}")

        init_mean = np.zeros((prior_transition.state_dimension,))

        init_mean[process_idcs[X_PROCESS_NUM]["state_d0"]] = sir_0
        init_mean[process_idcs[X_PROCESS_NUM]["state_d1"]] = init_sir_vel
        init_mean[process_idcs[X_PROCESS_NUM]["state_d2"]] = init_sir_vel

        # Set to inverse of link function
        init_mean[process_idcs[BETA_PROCESS_NUM]["state_d0"]] = beta_inverse_link_fn(beta_prior_mean)

        init_mean[process_idcs[BETA_PROCESS_NUM]["state_d1"]] = init_beta_vel

        # Cov
        sigma_sir = 0.001 * np.ones_like(sir_0)

        # Initialize the beta process at its stationary covariance
        stationary_beta_cov = scipy.linalg.solve_continuous_lyapunov(
            lf_transition.drift_matrix,
            -(lf_transition.dispersion_matrix @ lf_transition.dispersion_matrix.T),
        )
        sigma_beta = stationary_beta_cov[0, 0]
        sigma_velocity = 0.001

        init_marginal_vars = 1e-7 * np.ones((prior_transition.state_dimension,))
        init_marginal_vars[process_idcs[X_PROCESS_NUM]["state_d0"]] = sigma_sir
        init_marginal_vars[process_idcs[X_PROCESS_NUM]["state_d1"]] = sigma_velocity
        init_marginal_vars[process_idcs[X_PROCESS_NUM]["state_d2"]] = sigma_velocity

        init_marginal_vars[process_idcs[BETA_PROCESS_NUM]["state_d0"]] = sigma_beta
        init_marginal_vars[process_idcs[BETA_PROCESS_NUM]["state_d1"]] = sigma_velocity

        init_cov = np.diag(init_marginal_vars)

        initrv = randvars.Normal(init_mean, init_cov)
        
        time_domain = (0.0, float(num_covid_data_points + args.num_extrapolate))
        
        prior_process = randprocs.markov.MarkovProcess(
            transition=prior_transition, initrv=initrv, initarg=time_domain[0]
        )

        # Check jacobians

        _point = (
            prior_transition.proj2coord(proc=X_PROCESS_NUM, coord=0)
            @ prior_transition.proj2process(X_PROCESS_NUM)
            @ initrv.mean
        )
        _beta = np.array(0.3)
        _t = 0.1
        _m = initrv.mean

        ode_likelihood.check_jacobians(_t, _point, _beta, _m)

        # ##################################################################################
        # BUILD MODEL
        # ##################################################################################

        # ODE measurements
        measurement_matrix_ode = ode_measurement_cov * np.eye(STATE_DIM)
        measurement_noiserv_ode = randvars.Normal(mean=np.zeros(STATE_DIM), cov=measurement_matrix_ode)
        measurement_model_ode = randprocs.markov.discrete.NonlinearGaussian(
            input_dim=initrv.mean.size,
            output_dim=STATE_DIM,
            transition_fun=ode_likelihood.measure_ode,
            noise_fun=lambda t: measurement_noiserv_ode,
            transition_fun_jacobian=ode_likelihood.measure_ode_jacobian,
        )

        # EKF
        linearized_measurement_model_ode = filtsmooth.gaussian.approx.DiscreteEKFComponent(
            measurement_model_ode,
            forward_implementation=forward_implementation,
            backward_implementation=backward_implementation,
        )

        # Data measurements
        measurement_matrix_data = data_measurement_cov * np.eye(OBSERVATION_DIM)
        proj_state_to_S = (
            prior_transition.proj2coord(proc=X_PROCESS_NUM, coord=0)
            @ prior_transition.proj2process(X_PROCESS_NUM)
        )[0:1, :]

        measurement_noiserv_data = randvars.Normal(mean=np.zeros(OBSERVATION_DIM), cov=measurement_matrix_data)
        measurement_model_data = randprocs.markov.discrete.LTIGaussian(
            transition_matrix=proj_state_to_S,
            noise=measurement_noiserv_data,
            forward_implementation=forward_implementation,
            backward_implementation=backward_implementation,
        )

        # ##################################################################################
        # Run algorithm
        # ##################################################################################
        data_grid = np.array(train_idcs, copy=True, dtype=np.float64)
        ode_grid = np.arange(*time_domain, step=args.filter_step_size, dtype=np.float64)
        merged_locations = probssm.util.unions1d(data_grid, ode_grid)
        
        merged_observations = []
        merged_measmods = []

        data_idx = 0
        ode_idx = 0

        for loc in merged_locations:
            if np.in1d(loc, data_grid):
                merged_observations.append(SIR_data[data_idx, 0:1])
                merged_measmods.append(measurement_model_data)
                data_idx += 1

            elif np.in1d(loc, ode_grid):
                merged_observations.append(np.array(zero_data))
                merged_measmods.append(linearized_measurement_model_ode)
                ode_idx += 1
            else:
                pass

        merged_regression_problem = problems.TimeSeriesRegressionProblem(
            observations=merged_observations,
            locations=merged_locations,
            measurement_models=merged_measmods,
        )

        assert len(merged_observations) == len(merged_measmods) == len(merged_locations)

        kalman_filter = filtsmooth.gaussian.Kalman(prior_process)

        if args.filter_smoothing==1:
            logging.info("Computing smoothing posterior ...")
            start_filtsmooth = time.time()
            posterior, filtering_info_dicts = kalman_filter.filtsmooth(merged_regression_problem)
            time_filtsmooth = time.time() - start_filtsmooth

            logging.info(
                f"\033[1mFiltering + Smoothing took {time_filtsmooth:.2f} seconds.\033[0m"
            )

        else:
            logging.info("Computing filtering posterior ...")
            start_filter = time.time()
            posterior, filtering_info_dicts = kalman_filter.filter(merged_regression_problem)
            time_filter = time.time() - start_filter

            logging.info(
                f"\033[1mFiltering took {time_filter:.2f} seconds.\033[0m")


        means = np.stack([s.mean for s in posterior.states])
        covs = np.stack([s.cov for s in posterior.states])

        lik = 0
        data_idx = 0
        
        for i,loc in enumerate(merged_locations):
            if np.in1d(loc, data_grid):
                #lik+=np.exp(means[i,0]) - means[i,0] * np.exp(SIR_data[data_idx, 0])
                #lik+= np.abs(means[i,0] - SIR_data[data_idx,0])

                lik += stats.norm.logpdf(SIR_data[data_idx,0],
                                         loc=means[i,0],
                                         scale=np.sqrt(covs[i,0,0]))
                data_idx += 1
        #lik /= len(SIR_data[:,0])
        
        
        if args.num_samples is not None and args.num_samples > 0:
            logging.info(f"Drawing {args.num_samples} samples from posterior...")

            start_sampling = time.time()
            samples = posterior.sample(rng=rng, size=args.num_samples)
            time_sampling = time.time() - start_sampling

            _samples_save_file = log_dir / "posterior_samples.npy"
            np.save(_samples_save_file, samples)

            logging.info(f"Saved posterior samples to {_samples_save_file}.")

            logging.info(f"\033[1mSampling took {time_sampling:.2f} seconds.\033[0m")
        
        
        if save:
            _posterior_save_file = log_dir / "posterior_first.npz" 
            np.savez(_posterior_save_file,means=means,covs=covs)
            
            projections_dict = {
            "E_x": prior_transition.proj2process(X_PROCESS_NUM),
            "E_beta": prior_transition.proj2process(BETA_PROCESS_NUM),
            "E0_x": prior_transition.proj2coord(proc=X_PROCESS_NUM, coord=0),
            "E0_beta": prior_transition.proj2coord(proc=BETA_PROCESS_NUM, coord=0),
            }
            _projections_save_file = log_dir / "projections.npz"
            np.savez(_projections_save_file, **projections_dict)
            logging.info(f"Saved projections matrices to {_projections_save_file}.")

            data_dict = {
                "sir_data": np.exp(SIR_data),
                "day_zero": day_zero.to_numpy(),
                "date_range_x": np.array([ts.to_numpy() for ts in date_range_x]),
                "time_domain": np.array(time_domain),
                "data_grid": data_grid,
                "ode_grid": ode_grid,
                "dense_grid": merged_locations,
                "train_idcs": train_idcs,
                "val_idcs": val_idcs,
                "beta_prior_mean" : beta_prior_mean,
                "gamma" : gamma,
            }
            _data_info_save_file = log_dir / "data_info.npz"
            np.savez(_data_info_save_file, **data_dict)
            logging.info(f"Saved data info to {_data_info_save_file}.")

            _filtering_info_save_file = log_dir / "filtering_info.npz"
            np.savez(_filtering_info_save_file, filtering_info_dicts)
            logging.info(f"Saved filtering info to {_filtering_info_save_file}.")

            info = {}
            _args_save_file = log_dir / "info.json"
            probssm.args_to_json(_args_save_file, args=args, kwargs=info)
            logging.info(f"Saved info dict to {_args_save_file}.")
   
        return means, covs, lik
    
    #EM algorithm
    
    #hyperparameters
    assert np.isclose(args.filter_step_size, 1/24)
    step = 1+int(1/args.filter_step_size)
 
    beta_prior_mean = 0.1 #initial conditions for beta
    lengthscale = 72 # 72 hours = 3 * 24h = 3 days

    #parameters
    #R_cov = 1e-8 #data measurement cov initial parameter
    R_cov = 1e-6
    Q_cov = 1e-4
    
    gamma = 1/22.14 #gamma parameter in the ODE model
    
    #Useful functions  
    
    def transformed_states(arr):
        logS = arr[:, 0]
        S = np.exp(logS)
        logSp = arr[:, 1]
        Sp = S * logSp
        logI = arr[:, 3]
        I = np.exp(logI)
        logIp = arr[:, 4]
        Ip = I * logIp
        logR = arr[:, 6]
        R = np.exp(logR)
        logRp = arr[:, 7]
        Rp = R * logRp
        beta = arr[:, 9]
        x_offset = -scipy.special.logit(beta_prior_mean)
        y_offset = 0.0
        slope = args.sigmoid_slope
        beta_link = scipy.special.expit(slope * (beta - x_offset)) + y_offset
        return [S, Sp, I, Ip, R, Rp, beta]
    
    
    def opt_func(gamma, data, means):
        #Optimization function for gamma parameter optimization.
        start_idx = 150
        end_idx = 500
        logI = means[:, 3]
        I = np.exp(logI)
        logIp = means[:, 4]
        Ip = I * logIp
        beta = means[:, 9]
        data = np.exp(data[start_idx:end_idx, 0])
        
        x_offset = -scipy.special.logit(beta_prior_mean)
        y_offset = 0.0
        slope = args.sigmoid_slope
        
        beta_link = scipy.special.expit(slope * (beta - x_offset)) + y_offset
        
        S_gamma = (1000*(gamma*I + Ip)/(beta_link*I)).reshape(-1, step).mean(axis=1)[start_idx:end_idx]
        
        #Compute sqrt( sum ( (S - S(gamma))**2 ) )
        return np.sqrt( ( (data - S_gamma)**2 ).sum() )

    #beta_process_lengthscale = 75, beta_process_diffusion = 0.05, x_process_diffusion = 0.05, ode_measurement_cov = 5e-7, data_measurement_cov = 1e-9, gamma = 0.06, beta_prior_mean = 0.1, sir_0 = np.array(SIR_data[0, :3]), init_sir_vel = 1e-3, init_beta_vel = 0.0, save=False
    #function to call while optimizing
    
    f = lambda x1, x2, x3, save : get_states(data_measurement_cov = x1,
                                   gamma = x2,
                                   beta_process_diffusion=0.05,#0.01
                                   x_process_diffusion=0.05,#0.01
                                   ode_measurement_cov = x3,#5e-7
                                   beta_prior_mean = beta_prior_mean,
                                   beta_process_lengthscale = lengthscale,
                                   save = save)
    
    
    optim = True
    
    if optim:
        
        total_optim_step = 0
        while total_optim_step<1:
            
            #First, we optimize R
            logging.info(f"Initial R : {R_cov}")
            logging.info(f"Initial Q : {Q_cov}")
            logging.info(f"Initial gamma : {gamma}")
            
            for _ in range(3):
                
                for _ in range(4): #6 iterations is usually fine
                    #expectation step
                    means, covs, lik = f(R_cov, gamma, Q_cov, False)
                    #maximization step
                    #R_cov = ((SIR_data[:,0] - means[::step,0])**2).sum()/(len(SIR_data[:,0]) - 1) #MLE for R -> Maximizes likelihood
                    R_cov = (covs[::step, 0,0] + means[::step, 0]**2 - 2*SIR_data[:,0]*means[::step,0] + SIR_data[:,0]**2).mean()

                    logging.info(f"Current R : {R_cov}")
                    logging.info(f"Current Q : {Q_cov}")
                    logging.info(f"Current Log Likelihood : {lik}")



                #Second, we optimize Q
                for _ in range(4):
                    #expectation step
                    means, covs, lik = f(R_cov, gamma, Q_cov, False)

                    Q_cov = covs[:,[2,5,8],[2,5,8]].mean() #sample average of lambda from GP of x''(t)

                    logging.info(f"Current R : {R_cov}")
                    logging.info(f"Current Q : {Q_cov}")
                    logging.info(f"Current Log Likelihood : {lik}")
                
                
            '''
            #Second, we optimize gamma 
            for _ in range(10):
                #expectation step
                means, covs, lik = f(R_cov, gamma, Q_cov, False)
                
                #maximization step
                gamma_func = lambda x : f(R_cov, x, False)[2]
                #method 1 : Use Rp / I
                
                
                S, Sp, I, Ip, R, Rp, beta_link = transformed_states(means)
                gamma = np.median(Rp/I)
                logging.info(f"Current gamma : {gamma}")
                
                #gamma = (Rp * I).sum() / (I**2).sum()
                #method 2 : Use the likelihood via S(gamma)
                
                
                #gamma = minimize_scalar(gamma_func, bracket=[0.06, 0.07], method="brent").x
                
                #gamma = minimize_scalar(gamma_func,bracket=[0.05, 0.08], args=(SIR_data,means)).x #Maximize Likelihood w.r.t. gamma
                '''
                            
            total_optim_step+=1


    logging.info("Optimization completed.")
    
    #Run one more time and save the states for future analysis
    logging.info("Running algorithm with estimated parameters.")
    _, _, _ = f(R_cov, gamma, Q_cov, True)
    
    return


if __name__ == "__main__":
    main()
