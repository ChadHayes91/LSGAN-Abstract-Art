<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

# Project Description & Code

Link to Code:

## Introduction & Data
This project takes image classification using Convolutional Neural Networks (CNNs) explored in previous projects and expands upon that by creating a generative model - more specifically a Generative Adversarial Network (GAN) to create images. I found a dataset on Kaggle which webscraped around 3,000 abstract art images from WikiArt (https://www.wikiart.org/). 

Link to Dataset: https://www.kaggle.com/datasets/bryanb/abstract-art-gallery

## Methodology & Neural Net Architecture
GANs train two neural networks simultaneously, a generator network which takes as input a noise vector and creates an image, and a discriminator network which takes as input an image and determines if it is a fake image created by the generator or a real image from original dataset. For each batch of data, the discriminator analyzes both fake and real abstract art images and updates its weights from the loss via misclassification, and the generator updates its weights based on the loss from fake images the discriminator correctly classified as fake. This is similar to the minimax algorithm in classical AI where two players are paired off against each other.



## Difficulties & Model Improvements
GANs are typically tricky to train, there were a few difficulties I encoutered before I found reasonable results.

## Results

## Future Works & Improvements
