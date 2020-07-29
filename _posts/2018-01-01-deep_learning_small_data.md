# Model Selection with Large Neural Networks and Small Data
# Is the Curse of Dimensionality a Myth?


(IS the primary aim to validate the ranking hypothesis? Then experimental setting much simpler. And we are primarily interested in validating if we can do model/hyperparam tuning on small datasets!)

Highly overparameterized neural networks can display strong generalization performance, even on small datasets. 


This is certainly a bold claim, and I suspect many of you are shaking your heads right now.  

Under the classical teachings of statistical learning, this contradicts the well-known bias-variance tradeoff, i.e., increase model complexity at the expense of generalization error. 
This effect becomes even more pronounced for small datasets where the number of parameters, _p_, are larger than the number of observations, _n_. 


To address model selection in the small data domain using highly overparameterized neural networks, we review a recent ICML 2020 paper by [Deepmind](https://proceedings.icml.cc/static/paper_files/icml/2020/6899-Paper.pdf) (Bornschein, 2020) - henceforth named as "the paper".
The paper is an empirical study of generalization error as a function of training set size, which is clearly interesting from both a theoretical and practical viewpoint.
**Perhaps most useful** is the fact, that if we can **train on a smaller subset** of our training data while maintaining generalizable results, we can **reduce the computational overhead in model selection and hyperparmater tuning significantly**.  


And that is exactly the conclusion of the above paper and why we have chosen to review their method in this post.


# 1. Review of classical theory on the bias-variance trade-off
Before we get started, I will offer you two options. If you are tired of hearing about the bias-variance trade-off for the 100th time, please read the TLDR at the end of this Section 1 and then move on to Section 2. Otherwise, I will briefly introduce the bare minimum needed to understand the basics before moving on with the actual paper. 


The predictive error for all supervised learning algorithms can be broken into three parts, which are essential to understand the bias-variance tradeoff. These are;
1) Bias
2) Variance
3) Irreducible error (or noise term)

The **irreducible error** (sometimes called noise) is a term disconnected from the chosen model which can never be reduced. It is an aspect of the data arising due an imperfect framing of the problem, meaning we will never be able to capture the true relationship of the data no matter how good our model is.

The **bias** term is generally what people think of when they refer to model (predictive) errors. In short, it measures the difference between the "average" model prediction and the ground truth. Average might seem strange in this case as we typically only train one model, but think of it as the average of the range of predictions you get with the same model due to small pertubations or randomness in your data. **High bias** is a sign of poor model fit (underfitting), as it will have a large prediction error on both the training and test set.

Finally, the **variance** term refers to the variability of the model prediction for a given data point. It might sound similar, but the key difference lies in the "average" versus "data point". **High variance** implies high generalization error. For example, while a model might be relatively accurate on the training set, it achieves a considerably poor fit on the test set. This latter scenario (high variance, low bias) is typically the most likely when training overparameterized neural networks, i.e., what we refer to as **overfitting**.

You might have seen the typical dartboard for visualizing the four different combinations of these two terms. While it is a good idea to have solid intuition regarding these four scenarios, the practical implication implies balancing the bias and variance. This can be achieved in numerous ways, but the most popular ones are tuning the hyperparameters on a validation set and/or selecting a more complex/simple model. The ultimate goal is to obtain low bias and low variance. Though, in real-world scenarios, this is typically easier said than done, as reducing bias generally leads to an increase in variance, vice versa. 

As a final note, the bias-variance trade-off is purely a theoretical quantity, meaning we cannot quantify it in practice. Instead, we rely on the optimizing the balance between training, validation and test error.


Alright, I will assume you know enough about the bias-variance trade-off for now to understand why the original claim that **overparameterized neural networks do not ncessarily imply high variance**. 

## TLDR; high variance, low bias is a sign of overfitting. Overfitting happens when a model achieves high accuracy on the training set but low accuracy on the test set. This typically happens for overparameterized neural networks, especially on small datasets.

# Small Data, Big Decisions: Model Selection in the Small-Data Regime Paper Review

The long-held viewpoint regarding overfitting for overparamterized neural networks has been challenged recently by several researchers [insert citations].
One author (Belkin et al., 2019) even claims that a third regime exists besides underfitting and overfitting, which is called the _interpolation threshold_.
This regime  includes massively overparameterized models that you would expect to overfit the data excessively. Nevertheless, it has been shown to demonstrate gradual decline in generalization error following an initial peak. This scenario is termed _double-descent_, and it has also been empirically validated in Nakkiran et al., 2019 for modern neural network architectures on established and challenging datasets.

While there are many interesting hypothesis and empirical findings within this line of research, we will focus exclusively on the **relative ranking hypothesis**. Before we explain this hypothesis, we note that if proven true then **you** can potentially perform model selection and hyperparameter tuning on a small subset of your training dataset for your next experiment, and by doing so save computational resources and valuable training time. Due the this reason, we believe this hypothesis to be one of the most applicable and relevant for the majority of the people reading this post. 

As an additional experiment not included in the literature (as far as we know), we will investigate one setting that could potentially invalidate the relative ranking hypothesis, which is **imbalanced datasets**. 

Without further ado, lets try to break down the paper as efficiently as possible and include a few experiment.


## Ranking-hypothesis
The key hypothesis of the paper is; _"overparameterized model architectures seem to maintain their relative ranking in terms of generalization performance, when trained on arbitrarily small subsets of the training set"_. They call this observation the **relative ranking-hypothesis**. 

- Layman terms: Lets say we have 10 models to choose from, numbered from 1 to 10. We take a subset of our training data corresponding to 10% and find that model 6 is the best, followed by 4, then 3, and so on.. 

**The ranking hypothesis postulates, that as we gradually increase the subset percentage from 10% subset all the way up to 100%, we should obtain the exact same ordering of optimal models.** If this hypothesis is true, we can essentially perform model selection and hyperparamteter tuning on a small subset of the original data to the added benefit of much faster convergence. If this was not controversial enough, the authors even take it one step further as they found some experiments where training on small datasets led to more robust model selection (less variance), which certainly seem counterintuitive given that we would expect relatively more noise for smaller datasets.  

## Temperature calibration

One strange phenomenom of training neural network classifiers is that cross entropy error tends to increase even as classification error is reduced. This seems counterintuitive, but is simply due to models becoming overconfident in their predictions (Guo et al., 2017). We can use something called temperature scaling, which rectifies this overconfidence by calibrating the cross entropy estimates on a small held-out dataset. This yields more generalizeable and well-behaved results compared to classical cross-entropy, especially relevant for overparameterized neural networks. As a rough analogy, you can think of this as providing less "false negatives" regarding the number of overfitting cases.

While the authors do not provide explicit details on the exact softmax temperature calibration procedure, we use the following procedure;

- We define a held-out calibration dataset, C, equivalent to 10% of the training data.  
- We initialize temperature scalar to be xx.xx

- For each epoch;
1) Calculate cross-entropy loss on our calibration set C 
2) Optimize the temperature scalar using gradient descent on the calibration set [Guo et al., 2017](https://github.com/gpleiss/temperature_scaling) 
3) Use the updated temperature scalar to calibrate the regular cross entropy during gradient descent   
- After training for 50 epochs, we calculate the test error

That covers all the details of the paper. Lets turn to the experimental setting.

# Experiments


We start by briefly replicating the MNIST experiment and then turn to the imbalanced dataset case..

## MNIST 


We start by replicating the paper's study on MNIST, before moving on with the imbalanced dataset experiment. This is not meant to disprove any of the claims in the paper, but  simply to ensure we have replicated their experimental setup succesfully.

- Split of 90%/10% for the training and calibration sets, respectively
- Random sampling (as balanced subset sampling did not provide any added benefit according to the paper)
- 50 epochs
- Adam with fixed learning rate [10e-4] 
- Batch size = 256
- Fully connected MLPs with 3 hidden layers and 2048 units each
- With and without dropout 
- A simple convolutional network with 4 layers, 5x5 spatial kernel, stride 1 and 256 channels
- Logistic regression
- 10 different seeds to visualize uncertainty bands  (30 in original paper)

The authors also mention experimenting with replacing ReLU with tanh, batch-norm, layer-norm etc., but it is unclear if these tests were included in their final results. Thus, we only consider the experiment using the above settings. 


### Experiment 1: How does temperature scaling during gradient descent affect generalization?
As an initial experiment, we want to validate why temperature scaling is needed.
For this, we train an MLP using ReLU and 3 hidden layers of 2048 units each, respectively. We do not include dropout and we train for 50 epochs.

**Our hypothesis is:**: The test cross entropy should gradually increase while test accuracy decreases over time (motivation for temperature scaling in the first place, model overconfidence). 

**Here are the results from this initial experiment:**
![](/images/small_data_big_decisions/acc_vs_test_entropy_mean_no_scaling.png)
Clearly, the test entropy does decline initially and then gradually increase over time while test accuracy keeps improving. This is evidence in favor of hypothesis 1. Figure 3 in Guo et al., 2017, demonstrates the exact same effect on CIFAR-100. 
*Note:* We have smoothed the results a bit to make the effect more visible. 


**Conclusions from Experiment 1:**
- If we keep training large neural networks for sufficiently long, we start to see overconfident probabilistic predictions, making them less useful out-of-sample. 


To remedy this effect, we can incorporate temperature scaling which a) ensures probabilistic forecasts are more stable and reliable out-of-sample and b) improves generalization by scaling training cross entropy during gradient descent. This was shown in Guo et al., 2017.


### Relative Ranking Hypothesis 
Having shown that temperature scaling is needed, we now turn to the primary experiment - i.e., how does test cross-entropy vary as a function of the size of our training dataset.

We are essentially replicating Figure 5 on page 5 of the paper. Our results look as follows:



## Imbalanced Dataset
We will now conduct an experiment for the case of imbalanced datasets, which is not included in the actual paper, as it could be a setting where the tested hypothesis does not hold true. 

We use an artificially imbalanced MNIST dataset version from https://github.com/ufoym/imbalanced-dataset-sampler/blob/master/examples/mnist.ipynb. The distribution looks as follows:


We sample an artificially imbalanced version of MNIST similar to https://www.ijcai.org/Proceedings/2019/0334.pdf. 
The procedure is as follows. For each class in our dataset, we subsample it to be between 0 and 100 percent of the original training and testdataset. Then, we select our calibration dataset similar to the previous experiment, i.e., random 90/10% split between training and calibration.

We include a visualization of the classes distribution for the **original MNIST training dataset**

![](/images/small_data_big_decisions/distribution_original.png)


and the **imbalanced version**
![](/images/small_data_big_decisions/distribution_imbalanced.png)


While a plethora of different methods for overcoming the problem of imbalanced datasets exists, we want to investigate and isloate the effect of having an imbalanced dataset for the relative ranking hypothesis, i.e., can we still use model selection and hyperparameter tuning on smaller subsets of our original data in the imbalance data setting.
If the reader wants to review some of these different methods for mitigating the effect of imbalance on training neural networks, we kindly refer to https://arxiv.org/pdf/1710.05381.pdf.


After training our different models again for this imbalanced version, we obtain the following results:







# They also derive a term called the "Minimum Description Lengths" (MDL) for common datasets and modern neural network architectures. 
# MDL is inspired by the well-known Occam's razor principle, in which the model with the most simple

# - **Hypothesis 2:** If we turn on temperature scaling, we expect to see lower and more stable test cross entropy error during training (as demonstrated in Guo et al., 2017)

**Here are the results from our Hypothesis 2:**
