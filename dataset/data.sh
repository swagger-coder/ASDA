#!/bin/bash
# data process
python data_process.py --data_root ../ln_data --output_dir ../ln_data --dataset refcoco --split unc --generate_mask
python data_process.py --data_root ../ln_data --output_dir ../ln_data --dataset refcoco+ --split unc --generate_mask
python data_process.py --data_root ../ln_data --output_dir ../ln_data --dataset refcocog --split google --generate_mask
python data_process.py --data_root ../ln_data --output_dir ../ln_data --dataset refcocog --split umd --generate_mask

# datascript
python datascript.py --dataset refcoco
python datascript.py --dataset refcoco+ 
python datascript.py --dataset refcocog_google
python datascript.py --dataset refcocog_umd
