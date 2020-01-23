# Stochastic Video Generation with a Learned Prior

Today we will go through the paper [Stochastic Video Generation with a Learned Prior](https://arxiv.org/pdf/1802.07687.pdf) published in the Proceedings of the 35th International Conference on Machine Learning (ICML), PMLR 80:1174-1183, 2018.  
The accompanying code to the project can be found in the following official [Github repo](https://github.com/edenton/svg).  

## Motivation and Video Prediction
> *"Predicting future video frames is extremely challenging, as there are many factors of variation that make up the dynamics of how frames change through time"*  [[Villegas et al, 2019]](https://arxiv.org/abs/1911.01655) - NeurIPS 2019 Paper.  

Why is video prediction in the real-world such as hard problem?  

The main problem, as stated in the *Stochastic Video Generation with a Learned Prior* paper, is the **inherent uncertainty underlying the dynamics of the real-world**. For humans, the ability to anticipate the movement of objects and actions in videos is something that we are very good at. But for a neural network, however, the ability to capture all these spatio-temporal factors is an open problem to date.  

The majority of neural network-based video prediction so far have shown **increasingly blurry results** for multiple frame predictions in the future [ConvLSTM](https://papers.nips.cc/paper/5955-convolutional-lstm-network-a-machine-learning-approach-for-precipitation-nowcasting.pdf). This is due to the usage of a **deterministic loss function**, which implies objects are becoming increasingly blurry to **accomodate several possible futures**.  Some work has been done to overcome this by imposing distributions on the loss function, such as using an [adversarial loss function](https://arxiv.org/abs/1406.2661). Due to mode collapse and unstable training for video prediction tasks in particular, these methods not seen the same amount of success as for example [image generation models](https://arxiv.org/abs/1710.10196).  

This paper by Emily Denton and Rob Fergus propose a new method they call the **stochastic video generation (SVG)** model. Using this method, they are able to generate video frames **many steps into the future** that exhibit **high variability and are sharp**.

So how does the method actually work? The original paper is fairly technical, so we will try to dissect the various parts in this post in more detail.  

## Method
The specific method being used is a **deterministic video predictor with time-dependent stochastic latent variables**. 
Video prediction model that combines a deterministic prediction (model) of the next frame with stochastic latent variables, drawn from a time-varying distribution learned from training sequences (learned prior).

Two variants of this method exists with different priors; a) fixed prior over the latent variables (SVG-FP) and b) a learned prior (SVG-LP).   
So what does all these fancy words actually mean? 

The intuition behind this approach can be found on page 2 in the paper: "Intuitively, the latent variable $z_{t}$ carries all the stochastic information about the next frame that the deterministic model cannot capture. After conditioning on a short series of real frames, the model can generate multiple frames into the future by passing generated frames back into the input of the prediction model and, in the case of the SVG-LP model, the prior also".

At training time, training is guided by an inference model which estimates the latent distribution for each time step. Specifically, this model takes as input the generated frame from the prediction model x_t and also the previous frame $x_{1:t-1}$ and computes a distribution $q_{\phi}(z_{t}|x_{1:t})$ from which we sample $z_{t}$.
To ensure we do not simply replicate $x_{t}$ we use a KL-divergence term between $q_{\phi}(z_{t}|x_{1:t})$ and $p(z)$ to ensure they are not equivalent (i.e. capture new information not present in the previous frames).  

A second loss penalizes the reconstruction error between $\hat{x_{t}}$ and $x_{t}$.

We will now go through all these parts individually.

Skip connections between encoder and last ground truth frame - enabling the model to easily generate static background features.
### Deterministic Video Predictor

### Time-dependent stochastic latent variables

### Fixed prior

### Learned prior
As they state in the paper, the *"learned prior"* can be thought of as a *"predictive model of uncertainty"*. This essentially means, that for trajectory paths that are **easily predictable**, the learned prior predicts a **low uncertainty**. For **uncertain trajectories**, however, such as a ball hitting a surface, the **predicted uncertainty** will be **high**. 
![](/images/vae/vae_AMS.png)  

