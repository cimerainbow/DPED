# DPED
This repository contains the Implementation details of the paper "DPED: Bio-inspired dual-pathway network for edge detection".<br>
The address of the paper is at https://www.frontiersin.org/articles/10.3389/fbioe.2022.1008140/full <br>
If you have any questions, you can make issue or send email to [gauss.chenll@gmail.com](gauss.chenll@gmail.com). <br>

## Citations

If you are using the code/model/data provided here in a publication, please consider citing our paper:<br>
```
@article{chen2022dped,
  title={DPED: Bio-inspired dual-pathway network for edge detection},
  author={Chen, Yongliang and Lin, Chuan and Qiao, Yakun},
  journal={Frontiers in Bioengineering and Biotechnology},
  volume={10},
  year={2022},
  publisher={Frontiers Media SA}
}
```

## Get Start
All results is evaluated Python 3.7 with PyTorch 1.8.1 and MATLAB R2018b.<br>
You can run our model by following these steps:<br>

1. Download our code.
2. prepare the dataset.
3. Configure the environment.
4. If Windows (Linux) system, please modify the dataset_path in Win_cfgs.yaml(Lin_cfgs.yaml).
5. Run the "mian.py".

## Datsets
We use the links in [RCF](https://github.com/yun-liu/rcf) Repository (really thanks for that).<br>
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