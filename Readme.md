# Thinking Racial Bias in Fair Forgery Detection: Models, Datasets and Evaluations

[![Paper](https://img.shields.io/badge/arXiv-2407.14367v2-blue)](https://arxiv.org/abs/2407.14367v2)

![Main](fig/main.jpg)


## 🎁 Get our FairFD dataset

Please fill out this brief questionnaire [[Link]](https://docs.google.com/forms/d/e/1FAIpQLSdCAqk1olTdUci0S03KPDDzTrCElsvxJhCOQphAbbsZKGXiBA/viewform?usp=sf_link) to obtain our dataset and pretrained weights. We will timely return a link to download the content. 

After downloading the dataset and model weights, organize the directory structure as:
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

You need to bulid the conda or docker environment. You can build the environment following this [Link](https://github.com/SCLBD/DeepfakeBench?tab=readme-ov-file#-quick-start) to set up the environment. Do not forget to download pretrained weights in `./pretrained` folder. 

Download needed files and organize the directory structure as [Get our FairFD dataset](#-get-our-fairfd-dataset). 

### 2. Get confidence scores

Aftering installating well, you can run the following scripts to get the confidence scores of a provided model. 

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

Follow our step by step notebook `calculate_benchmark.ipynb` to calculate the final metrics based on previous confidence scores. 

## 🏆 Benchmark Results

![BenchmarkResults](fig/benchmark_results.jpg)

## 🎯 Bias Pruning with Fair Activations (BPFA)

We provide step by step notebooks to conduct pruning. 

```bash
# For normal models
pruning_pipeline.ipynb

# For fairness trained models
pruning_pipeline-fairness-enhanced.ipynb
```

## 🎯 New SOTA with our BPFA

![BPFA](fig/BPFA.jpg)


## 🏷️ Citation

If you have found FairFD or BPFA useful in your research, please consider citing the following paper:

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

In the implementation of this benchmark, we acknowledge the code and some weights from the repository: https://github.com/SCLBD/DeepfakeBench (NeruIPS 2023). 
