# Multi-Model Chorus Recognition
Basic code for paper: [TBD]  
The paper will be uploaded to arXiv after the anonymous period.  

## Dependency
- python 3.6+
- PyTorch 1.0+
- Transformers 2.6.0

- others
    - python_speech_features
    - Pandas

## Data
[TODO]

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
