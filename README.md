# Getting Started
## Neural Network
A hidden size of 70 and 2 layers yield good results: 
```
python train_hannds.py --hidden_size 70 --layers 2 --length 100 --cuda --cv_partition 1 --network 88 --rnn_type LSTM
```
Provide the `--cuda` flag to use the GPU.

## Kalman filter
```
python kalman_mapper.py
```

## Hannds server
```
python hannds-server.py
```
Then open simple-text.maxpat in Max/MSP.


    

