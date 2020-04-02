# Video Prediction using ConvLSTM Autoencoder (PyTorch)

In this guide, I will show you how to code a ConvLSTM autoencoder (seq2seq) model for frame prediction using the MovingMNIST dataset.
This framework can easily be extended for any other dataset as long as it complies with some standard configuration settings.

Before starting, we will briefly outline the libraries we are using and the steps we need to take:

```
pytorch
pytorch-lightning (for multi-GPU and easy/optimal configuration)
numpy
```

## Steps:
**1. Define dataloader**  
**2. Define model architecture**  
**3. Define main script for running training using pytorch-lightning**  


# 1: Dataloader
Download the dataloader script from the following repo [tychovdo/MovingMNIST](https://github.com/tychovdo/MovingMNIST)
This dataset was originally developed and described [here](http://www.cs.toronto.edu/~nitish/unsup_video.pdf), and it contains 10000 sequences each of length 20 with frame size 64 x 64 showing 2 digits moving in various trajectories (and overlapping).

Something to note beforehand is the inherent randomness of the digit trajectories. We do expect that this will become a major hurdle for the model we are about to describe, and we also note that newer approaches such as Variational Autoencoders might be a more efficient model for this type of task.


# 2: Model architecture
The specific model type we will be using is called a seq2seq model, which is typically used for NLP or time-series tasks (it was actually implemented in the Google Translate engine in 2016). 

The original papers on seq2seq are [Sutskever et al., 2014](https://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf) and [Cho et al., 2014](http://emnlp2014.org/papers/pdf/EMNLP2014179.pdf).

In its simplest case, the seq2seq model takes a sequence of items as input (such as words, word embeddings, letters, etc.) and outputs another sequence of items. For machine translation, the input could be a sequence of Spanish words and the output would be the English translation.

We can separate the seq2seq model into three parts, which are the a) encoder, b) encoder embedding vector and c) decoder. Lets write them out and describe each:

a) Encoder                  (encodes the input list)  
b) Encoder embedding vector (the final embedding of the entire input sequence)  
c) Decoder                  (decodes the embedding vector into the output sequence)  
  
Let's make it concrete with our machine translation example:

Encoder takes the Spanish sequence as input  
Encoder embedding vector is the final representation of our encoder  
Decoder outputs the English translation sequence.  

Hopefully part a) and part c) are somewhat clear to you. Arguably the most tricky part in terms of intuition for the seq2seq model is the encoder embedding vector. How do you define this vector exactly?  


## Meet RNN

Before you move any further, I highly recommend the following [excellent blog post](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) on RNN/LSTM. Understanding LSTM's intimately is an essential prerequisite for seq2seq models!  

So lets assume you fully understand what is a LSTM cell and (most importantly) what is the hidden state. Typically the encoder and decoder in seq2seq models consists of LSTM cells, such as the following figure:

![](/images/mnist_video_pred/encoder-decoder_2.png) 





Several extensions to the vanilla seq2seq model exists; the most notable being the [Attention module](https://arxiv.org/pdf/1409.0473.pdf).


## Frame Prediction
Frame prediction is inherently different from the original tasks of seq2seq such as machine translation. 
This is due to the fact, that RNN modules (LSTM) in the encoder and decoder use fully-connected layers to encode and decode word embeddings (which are represented as vectors). If you need a primer on LSTM, please read this [excellent blog post](https://colah.github.io/posts/2015-08-Understanding-LSTMs/).

Once we are dealing with frames we have 2D tensors, and to encode and decode these in a sequential nature we need an extension of the original seq2seq models using LSTMs.

This is where Convolutional LSTM (ConvLSTM) comes in. Presented at [NIPS in 2015](https://papers.nips.cc/paper/5955-convolutional-lstm-network-a-machine-learning-approach-for-precipitation-nowcasting.pdf), ConvLSTM modifies the inner workings of the LSTM mechanism to use convolutions instead of simple matrix multiplication.
This makes it suitable for processing images in a sequential nature and thus use it for frame prediction.

Given its strong modelling power in sequential tasks, we expect this model to perform well on frame prediction tasks such as the MovingMNIST dataset.


Lets write some code!

For our ConvLSTM implementation we use the implementation from the [CortexNet](https://arxiv.org/pdf/1706.02735.pdf) [ndrplz](https://raw.githubusercontent.com/ndrplz/ConvLSTM_pytorch/master/convlstm.py)

It looks as follows:

```python
import torch.nn as nn
import torch


class ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.

        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))




```

