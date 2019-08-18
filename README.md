## YOLO_v1_tensorflow



### Installation

1. Download [YOLO_small](https://pan.baidu.com/s/1mhE0WL6)
weight file and put it in `weights`

2. Modify configuration in `config.py`

3. Training
	```Shell
	$ python3 train.py
	```

4. Test
	```Shell
	$ python3 test.py
	```
    
### Tensorboard
For example:
    
    tensorboard --logdir=output/2019_08_17_18_38
    
### Environment
1. Tensorflow-gpu-1.9

2. Python 3.6.8
