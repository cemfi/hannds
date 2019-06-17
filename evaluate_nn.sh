#!/usr/bin/env bash

source activate neumus
python --version

python evaluate_nn.py --models_path models/1 > eval1.csv
python evaluate_nn.py --models_path models/2 > eval2.csv
python evaluate_nn.py --models_path models/3 > eval3.csv