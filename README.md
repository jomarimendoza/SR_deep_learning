# Super Resolution using Deep and Shallow Networks
Based on End-to-End Image Super-Resolution via Deep and Shallow Convolutional Networks

The concept of using deep and shallow networks is extracting high-frequency features and low-frequency features which are useful for reconstruction. This project improved the previously implemented network with modifications on feature extraction and reconstruction layers. 

## Requirements
1. python 3.6
2. keras 2 with tensorflow

## Deep Network
For the deep network, similar network architecture is implemented with modifications on feature extraction and reconstruction. A linear activation layer is inserted to map the reconstruction to the HR image. 
![](images/deep_network.jpg)

to train the data on deep network
'''
python eed_train.py
'''

## Shallow Network
The shallow network is improved by inserting a last layer with a linear activation.
![](images/shallow_network.jpg)

# References
[1] [End-to-End Image Super-Resolution via Deep and Shallow Convolutional Networks](https://arxiv.org/abs/1607.07680)

[2] [Enhanced Deep Residual Networks for Single Image Super-Resolution](https://arxiv.org/abs/1707.02921)

[3] [NTIRE 2017 Challenge on Single Image Super-Resolution: Methods and Results](http://personal.ie.cuhk.edu.hk/~ccloy/files/cvprw_2017_ntire.pdf)
