#!/bin/bash

python src/run_trials.py --dataset=ascadv1_fixed --seed-count=5 
python src/run_trials.py --dataset=ascadv1_variable --seed-count=5
python src/run_trials.py --dataset=aes_hd --seed-count=5
python src/run_trials.py --dataset=dpav4 --seed-count=5 
python src/run_trials.py --dataset=otiait --seed-count=5 
python src/run_trials.py --dataset=otp --seed-count=5 