# Model Selection with Large Neural Networks and Small Data
# Is the Curse of Dimensionality a Myth?

Highly overparameterized neural networks can display strong generalization performance, even on small datasets. 


This is certainly a bold claim and many of you are probably shaking your heads right now.  

Under the classical teachings of statistical learning, this contradicts the well-known bias-variance tradeoff, i.e., increase model complexity at the expense of generalization error. 
This scenario only becomes more outspoken for small datasets where the number of parameters, _p_, are larger than the number of observations, _n_. 

Nevertheless, this claim has been theoretically and empirically investigated by several researchers recently [insert citations]. 

In one recent paper (Belkin et al., 2019), the authors even argue that besides the traditional overfitting and underfitting regimes, there exist a third regime called the _interpolation threshold_.
This third regime includes massively overparameterized models, and is defined by a peak in generalization error followed by gradual decline.
This regime is termed _double-descent_, and it has also been empirically validated in Nakkiran et al., 2019, for modern neural network architecture on established and challenging datasets.


To address model selection in the small data domain using highly overparameterized neural networks, we review a recent ICML 2020 paper by [Deepmind](https://proceedings.icml.cc/static/paper_files/icml/2020/6899-Paper.pdf) (Bornschein, 2020).
The paper is an empirical study of generalization error as a function of training set size, which is obviously interesting from an academic point of view. 
But **perhaps even more useful** is the fact, that if we can **train on a smaller subset** of our training data while maintaining generalizable results, we can **reduce the computational overhead in model selection and hyperparmater tuning significantly**.
And that is exactly the conclusion of the above paper.  

In this article we will run through some of the key elements of this paper, and also include a few experiments to validate some of the claims from the paper.


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

## TLDR; high variance, low bias is a sign of overfitting, where the model obtains high accuracy on the training set but low accuracy on the test set. This typically happens for large and overparameterized neural networks, especially on small datasets. 

# Small Data, Big Decisions: Model Selection in the Small-Data Regime Paper Review

Let's briefly review the main takeaways from the "Small Data, Big Decisions: Model Selection in the Small-Data Regime Paper", followed by a few experiments to verify their claim that generalizability can be obtained using overparameterized models on small datasets, and if there are situations when this does not hold true.

Without further ado, lets try to break down the paper as efficiently as possible.

To reiterate the aim of the paper, it is an empirical investigation of generalization error as a function of training set size on various architectures and hyperparameters for both ImageNet, CIFAR10, MNIST and EMNIST. 


The key hypothesis of the paper is; _"overparameterized model architectures seem to maintain their relative ranking in terms of generalization performance, when trained on arbitrarily small subsets of the training set"_. They call this observation the ranking-hypothesis. Layman terms: Lets say we have 10 models to choose from, numbered from 1 to 10. We take a subset of our training data corresponding to 10% and find that model 6 is the best, followed by 4, then 3, and so on.. 
**The ranking hypothesis postulates, that as we gradually increase the subset percentage from 10% subset all the way up to 100%, we should obtain the exact same ordering of optimal models.** If this hypothesis is true, we can essentially perform model selection and hyperparamteter tuning on a small subset of the original data to the added benefit of much faster convergence. If this was not controversial enough, the authors even take it one step further as they found some experiments where training on small datasets led to more robust model selection (less variance), which certainly seem counterintuitive given that we would expect relatively more noise for smaller datasets.  

The final proposal is the usage of something called softmax-temperature, which should (in theory) yield more generalizeable and well-behaved results after being calbirated on a small held-out dataset compared to classical cross-entropy. As a rough analogy, you can think of this as providing less "false negatives" regarding the number of overfitting cases.

# MNIST Experiment

After that, we will conduct a few experiments of our own, specifically in the setting of imbalanced datasets, which is not included in the actual paper and could be a setting where the tested hypothesis does not hold true.

# Imbalanced Datasets Experiment





# They also derive a term called the "Minimum Description Lengths" (MDL) for common datasets and modern neural network architectures. 
# MDL is inspired by the well-known Occam's razor principle, in which the model with the most simple

# 
