# Getting Started
## Neural Network
A hidden size of 70 and 2 layers yield good results: 
```
python train_nn.py --hidden_size 70 --layers 2 --length 100
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


    

