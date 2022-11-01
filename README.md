# DPED
This repository contains the Implementation details of the paper "DPED: Bio-inspired dual-pathway network for edge detection".<br>

All results is evaluated Python 3.7 with PyTorch 1.8.1 and MATLAB R2018b.<br>
You can run our model by following these steps:<br>

1. Download our code.
2. prepare the dataset.
3. Configure the environment.
4. Run the "mian.py".

## Datsets
We use the links in RCF Repository (really thanks for that).<br>
The augmented BSDS500, PASCAL VOC, and NYUD datasets can be downloaded with:<br>
```
  wget http://mftp.mmcheng.net/liuyun/rcf/data/HED-BSDS.tar.gz
  wget http://mftp.mmcheng.net/liuyun/rcf/data/PASCAL.tar.gz
  wget http://mftp.mmcheng.net/liuyun/rcf/data/NYUD.tar.gz
```
Multicue Dataset is Here: <br>
```
  https://drive.google.com/file/d/1-tyt_KyzlYc9APafdh5mHJzh2K_F2hM8/view?usp=sharing
```


## Reference
When building our codeWe referenced the repositories as follow:<br>
1. [pidinet](https://github.com/cimerainbow/pidinet)
2. [RCF](https://github.com/yun-liu/rcf)
3. [HED Implementation](https://github.com/xwjabc/hed)
4. [DRC](https://github.com/cyj5030/DRC-Release)
5. [Timm](https://github.com/rwightman/pytorch-image-models)