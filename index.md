# Group Members
Riley Gamson (gamsoril), Eli Reed (erobreed)

# Introduction

## What is the problem?
Our final project was to identfiy and design a bird classification system which takes an image of a bird as input, and can identify its breed. We utilized the [Bird Data Set][https://www.kaggle.com/c/birds21sp/data] provided by the Kaggle competition for CSE 455 Spring 2021 to train our classifier. This dataset provides a set of 555 training image classes, and 10000 test images on which we can assess the accuracy of our classifier. We applied some of the techniques we learned this quarter to the problem, like transfer learning to speed up classifier training. Most of our models are primarily built on CNNs, but we also explored other network architectures. We used data augmentation transforms on the supplied data sets to try and mitigate overfitting. Part of our investigation was assessing how different models are constructed, and how the hyperparameters we used influenced their training.

## Datasets used
[Bird Data Set][https://www.kaggle.com/c/birds21sp/data]

## Pre-trained Models used
[ntsnet][https://pytorch.org/hub/nicolalandro_ntsnet-cub200_ntsnet/]. This implements the model described in "Learning to Navigate for Fine-grained Classification". It is trained on the CUB-200-2011 dataset.
[ResNet][https://pytorch.org/hub/pytorch_vision_resnet/]. This implements the deep learning network found in ["Deep Residual Learning for Image Recognition"][https://pytorch.org/hub/pytorch_vision_resnet/].


# Approach
## Techniques
* Convolution Neural Networks
* Transfer Learning
* Data Augmentation
* Multimodal Neural Networks
## Problems
First, any amount of training took an extraordinary amount of time. For this reason, we relied heavily on pre-trained models that we could generalize to our data set and save on training time. Tuning our hyperparameters took many repeated experiments before we could settle on what to use. We also found that one set of parameters did not adequately apply to both models that we used. There were also substantial challenges in training nts-net; using a more complex model was certainly a double-edged sword. While we were able to achieve pretty decent accuracy given the time that we had to spend doing transfer learning, the losses we saw during training were confusing at first, since they were consistently very high. 
## Why this approach is best
In doing research on pre-trained models, we learned about multimodal approaches like ntsnet which combine visual image data and natural language processing. Ntsnet is trained a dataset with 200 classes, but this dataset is augmented with natural language descriptors which are processed in parallel with visual data during training. This approach is likely best because the training dataset fuses extremely fine-grained information about the images during training that yield highly effective feature extraction and identfication. 


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
