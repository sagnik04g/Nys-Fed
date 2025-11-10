# Nys-Fed
A comprehensive study in Federated Learning to compare the Nyström Approximation with other Newton's method like FedNew, FedNS and DONE, along with first-order FL methods like FedAvg, FedProx and MOON. Our source code is implemented in PyTorch. The experiments are inspired by [FedNew](https://github.com/aelgabli/FedNew) and [FedNS](https://github.com/superlj666/FedNS), which we reimplemented in PyTorch from their original MATLAB versions.

## Prerequisites
1. Download the femnist and shakespeare dataset from LEAF
2. Download the cinic10 dataset from "https://www.kaggle.com/datasets/mengcius/cinic10"
3. Download the libsvm datasets "phishing, w8a and realsim" from "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/"

## Setup and Installation
Create a conda env and install the required libraries
```bash
conda create --name nysfed python=3.10
pip install -r requirements.txt
conda activate nysfed
```

## Run the experiments
1. Run the experiments for testing Nys-Fed on femnist and cinic10 with deep models like Resnet-18, Resnet-50 and a custom-built DNN with reduced paramaters residual model
```bash
bash experiments1.sh
```
2. Run the pytorch implementation of [FedNew](https://github.com/aelgabli/FedNew) and [FedNS](https://github.com/superlj666/FedNS) on w8a and phishing datasets
```bash
bash experiments2.sh
```
3. Run other experiments for comparison of Nys-Fed on w8a, phishing and realsim datasets as well as the shakespeare dataset
```bash
bash experiments3.sh
```
4. Run experiments with first-order FL methods on all datasets including femnist, cinic10, shakespeare, w8a, phishing and realsim.
```bash
bash experiments4.sh
```
5. Run experiments for comparing DONE method with Richardson's Iterations on w8a and phishing dataset.
```bash
bash experiments5.sh
```

#### Note: Keep the configuration same as kept in the bash files for running all the experiments.
