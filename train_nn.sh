#!/usr/bin/env bash

source activate pytorch
python --version

# length=( 10 20 40 60 100 150 200)
length=(40 60 100 150 200)
# layers=( 1 2 3 5 )
layers=( 2 3 5 )
for i in ${layers[@]}
do
    for j in ${length[@]}
    do
	    # echo "python train_nn.py --net RNN --layers $i --length $j"
	    # python train_nn.py --net RNN --layers $i --length $j
	    # echo ""
	    echo "python train_nn.py --net LSTM --layers $i --length $j"
	    python train_nn.py --net LSTM --layers $i --length $j
	done
done

sleep 10