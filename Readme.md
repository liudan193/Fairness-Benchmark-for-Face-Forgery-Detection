# Thinking Racial Bias in Fair Forgery Detection: Models, Datasets and Evaluations

[![Paper](https://img.shields.io/badge/arXiv-2407.14367-blue)](https://arxiv.org/abs/2407.14367)

![Main](fig/main.jpg)


This is the official project repository for the paper ["Thinking Racial Bias in Fair Forgery Detection: Models, Datasets and Evaluations"](https://arxiv.org/abs/2407.14367v2) (AAAI 2025). This paper proposes a fairness evaluation dataset for deepfake detection, along with a pruning-based method to enhance fairness.


## 🎁 Get our FairFD dataset

### ⭐⭐⭐ 1. Download FairFD dataset

First click this [link](https://mailstsinghuaeducn-my.sharepoint.com/:f:/g/personal/zq-wang24_mails_tsinghua_edu_cn/Ek17ILchHktAn_7qTwT13s4B7Ku6hI3JHRnqM2j0TweQpw?e=GNBK8j) to download our FairFD dataset. You will need a password to unzip the downloaded dataset. Please follow the steps below to obtain the password.

1. Since FairFD is built upon RFW, you should first obtain authorization from [RFW](http://whdeng.cn/RFW/testing.html). After that, please include the email you received from RFW in your email to us.
2. Our dataset also has a license, which you can read here: [license](./license.md).
3. Please email **Zongqi Wang (zq-wang24@mails.tsinghua.edu.cn)** to obtain the password. We will respond as soon as possible. Please ensure that your email is sent from a valid official (University or Company) account and includes the following information: 

```
Subject: Application to download the FairFD
Name: (your first and last name)
Affiliation: (University or Company where you work)
Department: (your department)
Position: (your job title)
Email: (must be the email at the above mentioned institution)

I have read and agree to the terms and conditions specified in the FairFD license.
The email content responsed by RFW in step1. 
```


### 2. Organize the directory structure

After downloading, organize the directory structure as follows:

```
dataset
├── test
|   ├── data
|   │   ├── African
|   │   │   └── *.jpg
|   │   ├── Asian
|   │   │   └── *.jpg
|   │   ├── Caucasian
|   │   │   └── *.jpg
|   │   ├── Indian
|   │   │   └── *.jpg
|   ├── FaceSwap
|   │   ├── African
|   │   │   └── *.jpg
|   │   ├── Asian
|   │   │   └── *.jpg
|   │   ├── Caucasian
|   │   │   └── *.jpg
|   │   ├── Indian
|   │   │   └── *.jpg
|   ├── ...
weights
├── xception.pth
├── ucf.pth
├── *.pth
Fairness-Benchmark-for-Face-Forgery-Detection (This repository)
└── code(*.py/.ipynb)
```

Note that `./dataset/test/data/` is the directory for real face.

## ⏳ Quick Start

### 1. Installation

You need to set up the Conda or Docker environment. You can build the environment by following this [Link](https://github.com/SCLBD/DeepfakeBench?tab=readme-ov-file#-quick-start). Please remember to download the pretrained weights into the [`./pretrained`](./pretrained) folder.

After downloading the necessary files, kindly organize the directory structure as outlined in [Get our FairFD dataset](#-get-our-fairfd-dataset).

### 2. Get confidence scores

After successful installation, you can run the following scripts to obtain the confidence scores for the provided model. 

```bash
# For normal models
python test-get-confidence.py \
--detector_path="./config/detector/xception.yaml" \
--weights_path="../weights/xception.pth"

# For fairness trained models
python test-get-confidence-fairness-enhanced.py \
--detector_path="./config/detector/pfgdfd.yaml" \
--weights_path="../weights/pfgdfd.pth"
```

### 3. Get benchmark results

Please follow our step-by-step notebook [`calculate_benchmark.ipynb`](calculate_benchmark.ipynb) to compute the final metrics based on the previous confidence scores.

## 🏆 Benchmark Results

![BenchmarkResults](fig/benchmark_results.jpg)

## 🎯 New SOTA with our Bias Pruning with Fair Activations (BPFA)

### Conduct pruning
We provide step by step notebooks [`pruning_pipeline.ipynb`](pruning_pipeline.ipynb) for normal models and [`pruning_pipeline-fairness-enhanced.ipynb`](pruning_pipeline-fairness-enhanced.ipynb) for fairness trained models to conduct pruning. 

### Results

![BPFA](fig/BPFA.jpg)


## 🏷️ Citation

If FairFD or BPFA is useful for your research, please consider citing the following paper: 

```bibtex
@article{liu2024thinking,
  title={Thinking Racial Bias in Fair Forgery Detection: Models, Datasets and Evaluations},
  author={Liu, Decheng and Wang, Zongqi and Peng, Chunlei and Wang, Nannan and Hu, Ruimin and Gao, Xinbo},
  journal={arXiv preprint arXiv:2407.14367},
  year={2024},
  url={https://arxiv.org/abs/2407.14367}
}
```

## Acknowledgement

We acknowledge the use of dataset, code and some weights from https://github.com/SCLBD/DeepfakeBench (NeruIPS 2023) and code from https://github.com/Purdue-M2/AI-Face-FairnessBench. If you cite our paper, please consider citing their paper as well:

1. Yan, Zhiyuan, Yong Zhang, Xinhang Yuan, Siwei Lyu, and Baoyuan Wu. "DeepfakeBench: a comprehensive benchmark of deepfake detection." In Proceedings of the 37th International Conference on Neural Information Processing Systems (NeurIPS), 2023.
2. Lin, Li, Xin Wang, and Shu Hu. "AI-Face: A Million-Scale Demographically Annotated AI-Generated Face Dataset and Fairness Benchmark." arXiv preprint arXiv:2406.00783 (2024).
3. Wang, Mei, Weihong Deng, Jiani Hu, Xunqiang Tao, and Yaohai Huang. "Racial faces in the wild: Reducing racial bias by information maximization adaptation network." In Proceedings of the ieee/cvf international conference on computer vision (ICCV), 2019.


