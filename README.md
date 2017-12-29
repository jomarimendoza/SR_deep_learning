# Super Resolution using Deep and Shallow Networks
Based on End-to-End Image Super-Resolution via Deep and Shallow Convolutional Networks [1]

The concept of using deep and shallow networks is extracting high-frequency features and low-frequency features which are useful for reconstruction. This project improved the previously implemented network with modifications on feature extraction and reconstruction layers. The network is only trained and tested on 4x scaling

## Requirements
- python 3.6
- keras 2 with tensorflow

### Data 
#### Training
[DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) dataset from [3] should be downloaded
```python
python create_data.py
```
**24000** train data and **3000** validation data for DIV2K is created

#### Test
Common datasets used for benchmarking are [**Set5**](https://uofi.box.com/shared/static/kfahv87nfe8ax910l85dksyl2q212voc.zip), [**Set14**](https://uofi.box.com/shared/static/igsnfieh4lz68l926l8xbklwsnnk8we9.zip) and [**BSD100**](https://uofi.box.com/shared/static/qgctsplb8txrksm9to9x01zfa4m61ngq.zip). The following datasets are obtained in https://github.com/jbhuang0604/SelfExSR. 

## Deep Network
For the deep network, similar network architecture is implemented with modifications on feature extraction and reconstruction. A linear activation layer is inserted to map the reconstruction to the HR image. 
![](images/deep_network.jpg)

to **train** and **test** the data on deep network
```python
python eed_train.py
```
```python
python eed_test.py # tested on Set14 images
```
The training converged at ###

## Shallow Network
The shallow network is improved by inserting a last convolution layer with a linear activation.
![](images/shallow_network.jpg)

to **train** and **test** the data on shallow network
```python
python ees_train.py
```
```python
python ees_test.py # tested on Set14 images
```
The training converged at epoch20, showing fast training of the shallow network.

## Deep and Shallow Network
Before combining both networks, the last linear activation layer is removed. The network is then followed by three residual block with an architecture as [2].
![](images/ds_network.jpg)

to **train** and **test** the data on deep and shallow network
```python
python eeds_train.py
```
```python
python eeds_test.py # tested on Set14 images
```
The training converged at ###



## Results
Analysis is measured using PSNR (peak signal-to-noise ratio) and SSIM (structural similarity index) (notation: PSNR/SSIM). The **average** of all the images is computed. Comparison to other methods are done by evaluating the images previously reconstructed in https://github.com/jbhuang0604/SelfExSR.

|  Dataset | Bicubic  |   NN   | ScSR     | SelfExSR | SRCNN  | A+ | EED | EES |
|:----------:|:--------:|:--------:|:------:|:------------:|:---------:|:--------:|:------:|:----:|
| **Set5**   |   33.64	|   35.78	|   36.24	| Sub-band	|   35.43	|   36.28	|    A+	|   36.50	| 
||
| **Set14**  |   30.39	|   31.34	|   32.30	| Sub-band	|   31.10	|   32.37	|    A+	|   32.62	|
||
| **BSD100** |   28.42	|   29.07	|   30.07	| Sub-band	|   28.84	|   30.08	|    A+	|   30.33	|


# References
[1] [End-to-End Image Super-Resolution via Deep and Shallow Convolutional Networks](https://arxiv.org/abs/1607.07680)

[2] [Enhanced Deep Residual Networks for Single Image Super-Resolution](https://arxiv.org/abs/1707.02921)

[3] [NTIRE 2017 Challenge on Single Image Super-Resolution: Methods and Results](http://personal.ie.cuhk.edu.hk/~ccloy/files/cvprw_2017_ntire.pdf)
