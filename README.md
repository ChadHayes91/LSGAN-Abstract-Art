<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

Link to Code (Jupyter Notebook):

## Introduction & Data

This project takes image classification using Convolutional Neural Networks (CNNs) explored in previous projects and expands upon that by creating a generative model - more specifically a Generative Adversarial Network (GAN) to create images. I found a dataset on Kaggle which webscraped around 3,000 abstract art images from WikiArt: [https://www.wikiart.org/](https://www.wikiart.org/).

Link to Dataset: [https://www.kaggle.com/datasets/bryanb/abstract-art-gallery/](https://www.kaggle.com/datasets/bryanb/abstract-art-gallery/)

## Methodology & Neural Net Architecture
GANs train two neural networks simultaneously, a generator network which takes as input a noise vector and creates an image, and a discriminator network which takes as input an image and determines if it is a fake image created by the generator or a real image from original dataset. For each batch of data, the discriminator analyzes both fake and real abstract art images and updates its weights from the loss via misclassification, and the generator updates its weights based on the loss from fake images the discriminator correctly classified as fake. This is similar to the minimax algorithm in classical AI where two players are paired off against each other.

Here are some examples from this dataset (these are the types of images the generator is trying to create):
<p align="center">
  <img width="500" height="500" src="https://github.com/ChadHayes91/LSGAN-Abstract-Art/blob/main/Images/Input_Examples_1.png?raw=true">
  <img width="500" height="500" src="https://github.com/ChadHayes91/LSGAN-Abstract-Art/blob/main/Images/Input_Examples_2.png?raw=true">
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

Similarly, I used the Adam optimizer with a learning rate of 0.0002 and betas=(0.5, 0.999) for the generator.

## Difficulties & Model Improvements
GANs are typically tricky to train, there were a few difficulties I encoutered before I found reasonable results.

The first issue I encountered was with my chosen architecture. I used some linear layers before convolutional layers for both the discriminator and generator which made my results significantly worse. My generator's loss was continually increasing over epochs, and the outputs from the generator did not look like my input images. Here is the plot of my losses and the output after 300 epochs:
<p align="center">
  <img width="500" height="500" src="https://github.com/ChadHayes91/LSGAN-Abstract-Art/blob/main/Images/Linear_BCE_300_Loss.png?raw=true">
  <img width="500" height="500" src="https://github.com/ChadHayes91/LSGAN-Abstract-Art/blob/main/Images/Linear_BCE_300.png?raw=true">
</p>

I iterated on my discriminator and generator architecture until I eventually found the architecture mentioned in the previous section which provided decent results. 

My next issue is with the loss function. I originally used Pytorch's binary cross entropy error to generate a loss. The outputs from using this loss function were:
<p align="center">
  <img width="500" height="500" src="https://github.com/ChadHayes91/LSGAN-Abstract-Art/blob/main/Images/UpdatedArchitecture_BCE_500.png?raw=true">
  <img width="500" height="500" src="https://github.com/ChadHayes91/LSGAN-Abstract-Art/blob/main/Images/UpdatedArchitecture_BSE_500_Loss.png?raw=true">
</p>

The results look reasonable, however there are a significant number of similar images. One of the issues with GANs is the generator can learn a template which reliably trick the discriminator and the generator can get stuck generating the same images over and over again. This is not desirable, a good generator should create a variety of images. I investigated this issue and discovered that changing the loss function resolved this problem. I implemented mean squared loss instead of binary cross entropy loss based on the results from this paper: [https://arxiv.org/pdf/1611.04076.pdf](https://arxiv.org/pdf/1611.04076.pdf).

## Results

After iterating on my network architecture and changing my loss function, my generator is able to create images like the following given an input vector of random noise:
<p align="center">
  <img width="500" height="500" src="https://github.com/ChadHayes91/LSGAN-Abstract-Art/blob/main/Images/Final_LSGAN_500_Gen_1.png?raw=true">
</p>

The plot of losses over epochs looks more reasonable as well. Rather than the generator experiencing increasing loss over time, the generator's loss seems to be stable and closer to the discriminator's loss.
<p align="center">
  <img width="500" height="500" src="https://github.com/ChadHayes91/LSGAN-Abstract-Art/blob/main/Images/Final_LSGAN_Loss.png?raw=true">
</p>
