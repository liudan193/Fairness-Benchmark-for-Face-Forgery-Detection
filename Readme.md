# Thinking Racial Bias in Fair Forgery Detection: Models, Datasets and Evaluations

[![Paper](https://img.shields.io/badge/arXiv-2407.14367v2-blue)](https://arxiv.org/abs/2407.14367v2)

![Main](fig/main.jpg)


This is the official project repository for the paper ["Thinking Racial Bias in Fair Forgery Detection: Models, Datasets and Evaluations"](https://arxiv.org/abs/2407.14367v2). 


## 🎁 Get our FairFD dataset

Please fill out this brief questionnaire [[Link]](https://docs.google.com/forms/d/e/1FAIpQLSdCAqk1olTdUci0S03KPDDzTrCElsvxJhCOQphAbbsZKGXiBA/viewform?usp=sf_link)  to access our dataset and pretrained weights. We will promptly provide a download link. After our paper is accepted, we will public the download link. 

After downloading, organize the directory structure as follows:

```
dataset
├── test
|   ├── FaceSwap
|   │   ├── African
|   │   │   └── *.jpg
|   │   ├── Asian
|   │   │   └── *.jpg
|   │   ├── Caucasian
|   │   │   └── *.jpg
|   │   ├── Indian
|   │   │   └── *.jpg
|   ├── SimSwap
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
  url={https://arxiv.org/abs/2407.14367v2}
}
```

## Acknowledgement

We acknowledge the use of code and some weights from https://github.com/SCLBD/DeepfakeBench (NeruIPS 2023) and code from https://github.com/Purdue-M2/AI-Face-FairnessBench. If you cite our paper, please consider citing their paper as well:

```bibtex
@inproceedings{DeepfakeBench_YAN_NEURIPS2023,
 author = {Yan, Zhiyuan and Zhang, Yong and Yuan, Xinhang and Lyu, Siwei and Wu, Baoyuan},
 booktitle = {Advances in Neural Information Processing Systems},
 editor = {A. Oh and T. Neumann and A. Globerson and K. Saenko and M. Hardt and S. Levine},
 pages = {4534--4565},
 publisher = {Curran Associates, Inc.},
 title = {DeepfakeBench: A Comprehensive Benchmark of Deepfake Detection},
 url = {https://proceedings.neurips.cc/paper_files/paper/2023/file/0e735e4b4f07de483cbe250130992726-Paper-Datasets_and_Benchmarks.pdf},
 volume = {36},
 year = {2023}
}

@article{lin2024aiface,
  title={AI-Face: A Million-Scale Demographically Annotated AI-Generated Face Dataset and Fairness Benchmark},
  author={Li Lin and Santosh and Xin Wang and Shu Hu},
  journal={arXiv preprint arXiv:2406.00783},
  year={2024}
}
```

