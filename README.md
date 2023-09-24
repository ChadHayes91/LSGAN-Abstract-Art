<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

Link to Code (Jupyter Notebook):

## Introduction & Data

This project takes image classification using Convolutional Neural Networks (CNNs) explored in previous projects and expands upon that by creating a generative model - more specifically a Generative Adversarial Network (GAN) to create images. I found a dataset on Kaggle which webscraped around 3,000 abstract art images from WikiArt: [https://www.wikiart.org/](https://www.wikiart.org/).

Link to Dataset: [https://www.kaggle.com/datasets/bryanb/abstract-art-gallery/](https://www.kaggle.com/datasets/bryanb/abstract-art-gallery/)

## Methodology & Neural Net Architecture
GANs train two neural networks simultaneously, a generator network which takes as input a noise vector and creates an image, and a discriminator network which takes as input an image and determines if it is a fake image created by the generator or a real image from original dataset. For each batch of data, the discriminator analyzes both fake and real abstract art images and updates its weights from the loss via misclassification, and the generator updates its weights based on the loss from fake images the discriminator correctly classified as fake. This is similar to the minimax algorithm in classical AI where two players are paired off against each other.

Here are some examples from this dataset:
<p align="center">
  <img width="500" height="350" src="https://github.com/ChadHayes91/LSGAN-Abstract-Art/blob/main/Images/Input_Examples_1.png?raw=true">
  <img width="500" height="350" src="https://github.com/ChadHayes91/LSGAN-Abstract-Art/blob/main/Images/Input_Examples_2.png?raw=true">
</p>

My chosen generator and discriminator architectures were inspired from this InfoGAN paper which explores different architectures for a variety of different datasets: [https://arxiv.org/pdf/1606.03657.pdf](https://arxiv.org/pdf/1606.03657.pdf) (see pages 12 and 13).

The final architecture for my discriminator is as follows:
<p align="center">
  <img width="500" height="500" src="https://github.com/ChadHayes91/LSGAN-Abstract-Art/blob/main/Images/Discriminator_Code.png?raw=true">
</p>

The final layer of flattening and then predicting a real or fake classification is to use the squared error loss (instead of using Sigmoid for a BCE loss). I used the Adam optimizer for the discriminator with a learning rate of 0.0002 and betas=(0.5, 0.999).

The final architecture for my generator is as follows:
<p align="center">
  <img width="500" height="500" src="https://github.com/ChadHayes91/LSGAN-Abstract-Art/blob/main/Images/Generator_Code.png?raw=true">
</p>

Similarly, I used the Adam optimizer with a learning rate of 0.0002 and betas=(0.5, 0.999) for the generator. The loss for the generator is computed by correct classifications of fake images from the discriminator.

## Difficulties & Model Improvements
GANs are typically tricky to train, there were a few difficulties I encoutered before I found reasonable results.

The first issue I encountered was with my chosen architecture. I used some linear layers before convolutional layers for both the discriminator and generator which made my results significantly worse. My generator's loss was continually increasing over epochs, and the outputs from the generator did not look like my input images. Here is the plot of my losses and the output after 300 epochs:
<p align="center">
  <img width="500" height="500" src="https://github.com/ChadHayes91/LSGAN-Abstract-Art/blob/main/Images/Linear_BCE_300_Loss.png?raw=true">
  <img width="500" height="500" src="https://github.com/ChadHayes91/LSGAN-Abstract-Art/blob/main/Images/Linear_BCE_300.png?raw=true">
</p>


Paper desribing using least squared loss for optimizting GANs isntead of BCE: [https://arxiv.org/pdf/1611.04076.pdf](https://arxiv.org/pdf/1611.04076.pdf)

## Results

## Future Works & Improvements
