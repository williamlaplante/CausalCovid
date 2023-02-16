#!/bin/sh

case "$1" in
    "ger")  log-sirs-run @common_args.txt \
            --logdir "./run_$1" \
            --country "Germany"
        ;;
    "can") log-sirs-run @common_args.txt \
           --logdir "./run_$1" \
           --country "Canada"
        ;;
    *)  echo "This sub-experiment does not exist yet";;
esac