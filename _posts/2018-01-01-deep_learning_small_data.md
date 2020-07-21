# Model Selection with Large Neural Networks and Small Data
# Is the Curse of Dimensionality a Myth?

Highly overparameterized neural networks can display strong generalization performance, even on small datasets. 


This is certainly a bold claim and many of you are probably shaking your heads right now.  

Under the classical teachings of statistical learning, this contradicts the well-known bias-variance tradeoff, i.e., increase model complexity at the expense of generalization error. 
This scenario only becomes more outspoken for small datasets where the number of parameters, _p_, are larger than the number of observations, _n_. 

Nevertheless, this claim has been theoretically and empirically investigated by several researchers recently [insert citations]. 

In one recent paper (Belkin et al., 2019), the authors even argue that besides the traditional overfitting and underfitting regimes, there exist another regime called the _interpolation threshold_.
This third regime includes massively overparameterized models, and is defined by a peak in generalization error followed by gradual decline.
This regime is termed _double-descent_, and it has also been empirically validated in Nakkiran et al., 2019, for modern neural network architecture on established and challenging datasets.


To address model selection in the small data domain using highly overparameterized neural networks, we review a recent ICML 2020 paper by [Deepmind](https://proceedings.icml.cc/static/paper_files/icml/2020/6899-Paper.pdf).
The paper is an empirical study of generalization error as a function of training set size, which is obviously interesting from an academic point of view. 
But perhaps even more useful is the fact, that if we can train on a smaller subset of our training data while still obtaining generalizable results, we can reduce the computational overhead in model selection and hyperparmater tuning significantly. 
And that is exactly the conclusion of the above paper.

# They also derive a term called the "Minimum Description Lengths" (MDL) for common datasets and modern neural network architectures. 
# MDL is inspired by the well-known Occam's razor principle, in which the model with the most simple

# 
