#!/bin/bash

conda activate kg_env_new

./GnnExplainer.py $1
#./GnnExplainer.py 'spouse' 
#./GnnExplainer.py 'uncle'
#./GnnExplainer.py 'aunt' 
#./GnnExplainer.py 'successor'
#./GnnExplainer.py 'predecessor' 
#./GnnExplainer.py 'grandparent' 
#./GnnExplainer.py 'full_data'