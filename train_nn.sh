#!/usr/bin/env bash

source activate pytorch
python --version

layers=( 2 3 )
hidden=( 2 5 10 20 50 70 100 )
for i in ${hidden[@]}
do
    for j in ${layers[@]}
    do
	    echo "train_nn.py --hidden_size $i --layers $j --length 100 --cuda"
	    python train_nn.py --hidden_size $i --layers $j --length 100 --cuda
	done
done

sleep 10