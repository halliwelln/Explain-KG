#!/bin/bash

conda activate kg_env_new

#./GnnExplainer.py $1 $2 $3 $4 $5
./GnnExplainer.py royalty_30k full_data 10 25 0.001
