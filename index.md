# Introduction
## What is the problem?
Our final project was to design a bird classification system which takes an image of a bird as input, and can identify its breed. We utilized the [Bird Data Set][https://www.kaggle.com/c/birds21sp/data] provided by the corresponding kaggle competition for CSE 455 Spring 2021 to train our classified. This dataset provides a set of 555 training image classes, and 10000 test images on which we can assess the accuracy of our classifier. We relied on concepts we developed in class, specifically convolutional neural networks and transfer learning. We also wanted to examine hyperparameter optimization to see if that could significantly improve the functionality of our model. Finally, we used data augmentation to prevent overfitting. We attempted the kaggle challenge using three different pre-trained neural nets, listed below. Each provides a different architecture, and we wanted to identify how effectively each could extract features from our 
## Datasets used
[Bird Data Set][https://www.kaggle.com/c/birds21sp/data]

## Pre-trained Models used
[ntsnet][https://pytorch.org/hub/nicolalandro_ntsnet-cub200_ntsnet/]. This implements the model described in "Learning to Navigate for Fine-grained Classification". It is trained on the CUB-200-2011 dataset.
[Resnet][https://pytorch.org/hub/pytorch_vision_resnet/]. This implements the deep learning network found in ["Deep Residual Learning for Image Recognition"][https://pytorch.org/hub/pytorch_vision_resnet/].


# Approach
1. We started with resnet first, in order to develop some understanding of how changing the hyperparameters would influence the trained model. 
## Techniques
* Convolution Neural Network
* Transfer Learning
* Data Augmentation
## Problems
* One of the biggest issues we faced was 
## Why this approach is best


# Experiments
## Resnet
## ntsnet

# Results
## Resnet
### Data
### Analysis
## ntsnet
### Data
### Analysis

# Discussion
## What worked well/what didn't, and why?
## Did you learn anything?
## Can anything in this project be applied more broadly?

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://docs.github.com/categories/github-pages-basics/) or [contact support](https://support.github.com/contact) and we’ll help you sort it out.
