# AlignRE
This repository contains the official implementation for the paper: **[AlignRE: An Encoding and Semantic Alignment Approach for Zero-Shot  Relation Extraction](https://aclanthology.org/2024.findings-acl.174.pdf).** The paper has been accepted to appear at **ACL 2024 Findings**.

## Dataset

- [FewRel (Xu et al., 2018)](https://drive.google.com/file/d/1PgSTaEEUxsE-9lhQan3Yj91pzLhxv7cT)
- [WikiZSL (Daniil Sorokin and Iryna Gurevych, 2017)](https://drive.google.com/file/d/1kGmhlpTTq8UmIUPZ2CSIruWWsi_l_ERH)

Place them to the `/data` folder.

## Requirements

The main requirements are:

- python==3.9.7
- pytorch==1.13.1
- transformers==4.44.2
- sentence-transformers==2.2.2
- numpy==1.22.3

## Run

1. Download the pretrained [Bert weights](https://huggingface.co/bert-base-uncased) to folder './BERT_MODELS/bert-base-uncased'.
2. Download the pretrained [Sentence-Bert weights](https://huggingface.co/sentence-transformers/stsb-bert-base) to folder './BERT_MODELS/stsb-bert-base'.
3. Set hyperparameters and run `train.py`

## Cite

```latex
@inproceedings{li2024alignre,
  title={AlignRE: An Encoding and Semantic Alignment Approach for Zero-Shot Relation Extraction},
  author={Li, Zehan and Zhang, Fu and Cheng, Jingwei},
  booktitle={Findings of the Association for Computational Linguistics ACL 2024},
  pages={2957--2966},
  year={2024}
}
```

## Acknowledgement

The framework of AlignRE is based on [ZS-BERT](https://github.com/dinobby/ZS-BERT) and [RE-Matching](https://github.com/zweny/RE-Matching). Their contributions have greatly helped in the development and implementation of our code. We appreciate their efforts and the open-source community for fostering collaboration and knowledge sharing.
