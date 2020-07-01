# Stochastic Video Generation with a Learned Prior

Today we will go through the paper [Stochastic Video Generation with a Learned Prior](https://arxiv.org/pdf/1802.07687.pdf) published in the Proceedings of the 35th International Conference on Machine Learning (ICML), PMLR 80:1174-1183, 2018.  
The accompanying code to the project can be found in the following official [Github repo](https://github.com/edenton/svg).  

## Motivation and Video Prediction
> *"Predicting future video frames is extremely challenging, as there are many factors of variation that make up the dynamics of how frames change through time"*  [[Villegas et al, 2019]](https://arxiv.org/abs/1911.01655) - NeurIPS 2019 Paper.  

Why is video prediction in the real-world such as hard problem?  

The main problem, as stated in the *Stochastic Video Generation with a Learned Prior* paper, is the **inherent uncertainty underlying the dynamics of the real-world**. For humans, the ability to anticipate the movement of objects and actions in videos is something that we are very good at. But for a neural network, however, the ability to capture all these spatio-temporal factors is non-trivial and still an open problem to date.  

The majority of neural network-based video prediction so far have shown **increasingly blurry results** for multiple frame predictions in the future [ConvLSTM](https://papers.nips.cc/paper/5955-convolutional-lstm-network-a-machine-learning-approach-for-precipitation-nowcasting.pdf). This is due to the usage of a **deterministic loss function**, which implies objects are becoming increasingly blurry to **accomodate several possible futures**. It is worth mentioning, that newer methods such as the PredRNN, PredRNN+ and Eidetic 3D-LSTM also improve considerably over the baseline ConvLSTM model, but these methods are also solely focused on reducing this deterministic loss. 

Some work has been done to overcome this by imposing distributions on the loss function, such as using an [adversarial loss function](https://arxiv.org/abs/1406.2661). Due to mode collapse and unstable training for video prediction tasks in particular, these methods not seen the same amount of success as for example [image generation models](https://arxiv.org/abs/1710.10196). 

The paper by Emily Denton and Rob Fergus propose a new method called the **stochastic video generation (SVG)** model. Using this method, they are able to generate video frames **many steps into the future** that exhibit **high variability and are sharp**. 

So how does the method actually work? The original paper is fairly technical, so we will try to dissect the various parts in this post in more detail.  

## Method
The method is labelled (somewhat generically) as a **deterministic video predictor with time-dependent stochastic latent variables**. This implies, that we can split the method into two; 1) deterministic prediction model and 2) time-dependent stochastic latent variables drawn from some distribution (that we will come to shortly).

To the attentative reader, it should be clear that this method is analogous to variational autoencoders (VAEs), where we sample a prior from some specific distribution (typically Gaussian) and use this for image generation for example (in contrast to GANs where). Before moving further, lets briefly discuss variational auto-encoders (see this post for a great introduction if you are unfamiliar with VAE)  https://towardsdatascience.com/understanding-variational-autoencoders-vaes-f70510919f73).

### VAE
One of the first question that comes to mind when starting to study variational autoencoders is; why do we not simply use a regular autoencoder to train and optimize reconstruction loss on any given dataset, and then sample from the latent space to generate new images? (note: latent space here refers to the final encoding of the input before being fed to the decoder, see image xxx)

The problem is as follows. Since the autoencoder has been specifically trained for encoding and decoding all input data as lossless as possible, it does not impose any structure or regularization on the latent space. This essentially means, that there are no limits on how much the autoencoder can overfit the training data (given sufficient model capacity of course). This overfitting directly implies that sampling from the latent space will not work well for novel content generation such as images. Simple example: Say we have a dataset of a dog, a cat and a human. Training an autoencoder to reduce the reconstruction loss for these three images, the model could learn an explicit mapping that only works for these three examples individually, i.e., how do you compress and decompress the image of this specific cat versus this specific dog etc. The chances of sampling from the latent space of this model and generate something new such as a cow is extremely low.

This leads to variational autoencoders, where we specifically impose regularization in order to avoid the problems mentioned above. Using the definition from the above article; **"a variational autoencoder can be defined as being an autoencoder whose training is regularised to avoid overfitting and ensure that the latent space has good properties that enable generative process."**. 

The reguralisation is specifically done on the latent space. However, this regularsation is not simply achived via some regularization term or dropout, etc., which might be the first things that come to mind for a machine learning practitioneer. Rather, we slighlty alter the entire encoding and decoding process to go from a deterministic model (regular autoencoder) to a probablistic model (variational autoencoder). Compare as follows:   

**Regular autoencoder:**  
For a given input x - we encode a latent representation z=e(x) - and finally reconstruct our input in the decoder, d(z)  

**Variational autoencoder:**  
For a given input x - we encode a latent **distribution** p(z|g) - then we sample from this latent representation \mathbf{z} \sim \mathbf{p}(\mathbf{z} \mid \mathbf{x}) - and finally reconstruct our input in the decoder, d(z).  


Why go from deterministic to probabilstic?  
The reason is that we can directly express (and calculate a loss for) the latent space regularisation, rather than just having a single point reconstruction loss. In practice, the latent distribution is typically a Gaussian since we can represent it via the mean and covariance matrix. The regularisation term (or loss) then becomes the difference between the encoded distribution and a standard Gaussian, which can be expressed using the Kullback-Leibler (KL) divergence. **Minimizing this loss effectively means, that we train our model to encode the latent distribution to be close to a standard normal distribution, thus achieving a regularized latent space.**  Obviously only applying the regularization loss would be nonsensical (decoder would not play any role), meaning that we still impose the reconstruction loss upon our entire encoding-decoding scheme.

In short:  
a) Reconstruction loss enables our model to learn an effective encoding and decoding of data
b) Regularization loss imposes regularisation upon our latent space  


Their method is analogous tovariational autoencoders since a latent variable is sam-pled from a prior (in this case a learned prior).  After-wards, the latent variable is combined with an encodingof the previous frames to generate future frames that in-cludes time-varying degrees of uncertain


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

