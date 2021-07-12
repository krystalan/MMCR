# Multi-Model Chorus Recognition
Code and Dataset of ICANN2021 paper: *Multi-Model Chorus Recognition for Improving Song Search* 

## Dependency
- python 3.6+
- PyTorch 1.0+
- Transformers 2.6.0

- others
    - python_speech_features
    - Pandas

## CHORD Dataset
Coming soon!

## Pre-trained Language Model
We used  ```BERT-wwm-ext, Chinese``` pre-trained language model  
Related introduction and download link please refer to <u>[Chinese-BERT-wwm](https://github.com/ymcui/Chinese-BERT-wwm#%E4%B8%AD%E6%96%87%E6%A8%A1%E5%9E%8B%E4%B8%8B%E8%BD%BD)</u>.

## Train and Test
For training, you can run commands like this:  
```shell
python train.py
```

For evaluation, the command may like this:
```shell
python test.py
```

## Citation
If you use this code useful, please star our repo or consider citing:
```
@article{Wang2021MultiModalCR,
  title={Multi-Modal Chorus Recognition for Improving Song Search},
  author={Jiaan Wang and Zhixu Li and Binbin Gu and Tingyi Zhang and Qingsheng Liu and Zhigang Chen},
  journal={ArXiv},
  year={2021},
  volume={abs/2106.16153}
}
```
