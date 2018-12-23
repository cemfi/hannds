#!/usr/bin/env bash

source activate pytorch
python --version

start=1
end=10
for ((i=start; i<=end; i++))
do
    python train_nn.py --hidden_size 70 --layers 2 --length 100 --cuda --cv_partition $i --network MIDI --rnn_type GRU --bidirectional
    python train_nn.py --hidden_size 70 --layers 2 --length 100 --cuda --cv_partition $i --network MIDI --rnn_type LSTM --bidirectional
    python train_nn.py --hidden_size 70 --layers 2 --length 100 --cuda --cv_partition $i --network Magenta --rnn_type GRU --bidirectional
    python train_nn.py --hidden_size 70 --layers 2 --length 100 --cuda --cv_partition $i --network Magenta --rnn_type LSTM --bidirectional
    python train_nn.py --hidden_size 70 --layers 2 --length 100 --cuda --cv_partition $i --network MIDI --rnn_type GRU
    python train_nn.py --hidden_size 70 --layers 2 --length 100 --cuda --cv_partition $i --network MIDI --rnn_type LSTM
    python train_nn.py --hidden_size 70 --layers 2 --length 100 --cuda --cv_partition $i --network Magenta --rnn_type GRU
    python train_nn.py --hidden_size 70 --layers 2 --length 100 --cuda --cv_partition $i --network Magenta --rnn_type LSTM
done
wait
sleep 10