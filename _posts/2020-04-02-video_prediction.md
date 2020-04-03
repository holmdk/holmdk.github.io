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
Download the dataloader script from the following repo [tychovdo/MovingMNIST](https://github.com/tychovdo/MovingMNIST).  
This dataset was originally developed and described [here](http://www.cs.toronto.edu/~nitish/unsup_video.pdf), and it contains 10000 sequences each of length 20 with frame size 64 x 64 showing 2 digits moving in various trajectories (and overlapping).

Something to note beforehand is the inherent randomness of the digit trajectories. We do expect that this will become a major hurdle for the model we are about to describe, and we also note that newer approaches such as Variational Autoencoders might be a more efficient model for this type of task.


# 2: Model architecture
The specific model type we will be using is called a seq2seq model, which is typically used for NLP or time-series tasks (it was actually implemented in the Google Translate engine in 2016). 

The original papers on seq2seq are [Sutskever et al., 2014](https://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf) and [Cho et al., 2014](http://emnlp2014.org/papers/pdf/EMNLP2014179.pdf).

In its simplest configuration, the seq2seq model takes a sequence of items as input (such as words, word embeddings, letters, etc.) and outputs another sequence of items. For machine translation, the input could be a sequence of Spanish words and the output would be the English translation.

We can separate the seq2seq model into three parts, which are the a) encoder, b) encoder embedding vector and c) decoder. Lets write them out and describe each:

a) Encoder                  (encodes the input list)  
b) Encoder embedding vector (the final embedding of the entire input sequence)  
c) Decoder                  (decodes the embedding vector into the output sequence)  
  
For our machine translation example, this would mean:

- Encoder takes the Spanish sequence as input by processing each word sequentially
- The encoder outputs an embedding vector as the final representation of our input  
- Decoder takes the embedding vector as input and then outputs the English translation sequence  

**Hopefully part a) and part c) are somewhat clear to you**. Arguably the most tricky part in terms of intuition for the seq2seq model is the encoder embedding vector. How do you define this vector exactly?  


## Meet RNN

Before you move any further, I highly recommend the following [excellent blog post](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) on RNN/LSTM. Understanding LSTM's intimately is an essential prerequisite for most seq2seq models!  

Here are the equations for the regular LSTM cell:  

\begin{equation}i_{t}=\sigma\left(W_{x i} x_{t}+W_{h i} h_{t-1}+W_{c i} \circ c_{t-1}+b_{i}\right)\end{equation}
\begin{equation}f_{t}=\sigma\left(W_{x f} x_{t}+W_{h f} h_{t-1}+W_{c f} \circ c_{t-1}+b_{f}\right)\end{equation}
\begin{equation}c_{t}=f_{t} \circ c_{t-1}+i_{t} \circ \tanh \left(W_{x c} x_{t}+W_{h c} h_{t-1}+b_{c}\right)\end{equation}
\begin{equation}o_{t}=\sigma\left(W_{x o} x_{t}+W_{h o} h_{t-1}+W_{c o} \circ c_{t}+b_{o}\right)\end{equation}
\begin{equation}h_{t}=o_{t} \circ \tanh \left(c_{t}\right)\end{equation}
where $\circ$ denotes the Hadamard product.  

So lets assume you fully understand what a LSTM cell is and how cell states and hidden states work. Typically the encoder and decoder in seq2seq models consists of LSTM cells, such as the following figure:

![](/images/mnist_video_pred/encoder-decoder_2.png) 
### Breakdown
- The LSTM Encoder consists of 4 LSTM cells and the LSTM Decoder consists of 4 LSTM cells.
- Each input (word or word embedding) is fed into a new encoder LSTM cell together with the hidden state (output) from the previous LSTM cell
- The hidden state from the final LSTM encoder cell is (typically) the Encoder embedding. It can also be the entire sequence of hidden states from all encoder LSTM cells (note - this is not the same as attention)
- The LSTM decoder uses the encoder state(s) as input and procceses these iteratively through the various LSTM cells to produce the output. This can be unidirectional or bidirectional

Several extensions to the vanilla seq2seq model exists; the most notable being the [Attention module](https://arxiv.org/pdf/1409.0473.pdf).

Having discussed the seq2seq model, lets turn our attention to the task of frame prediction!

## Frame Prediction
Frame prediction is inherently different from the original tasks of seq2seq such as machine translation. 
This is due to the fact, that RNN modules (LSTM) in the encoder and decoder use fully-connected layers to encode and decode word embeddings (which are represented as vectors). 

Once we are dealing with frames we have 2D tensors, and to encode and decode these in a sequential nature we need an extension of the original LSTM seq2seq models.

### ConvLSTM
This is where Convolutional LSTM (ConvLSTM) comes in. Presented at [NIPS in 2015](https://papers.nips.cc/paper/5955-convolutional-lstm-network-a-machine-learning-approach-for-precipitation-nowcasting.pdf), ConvLSTM modifies the inner workings of the LSTM mechanism to use the convolution operation instead of simple matrix multiplication. Lets write our new equations for the ConvLSTM cells:  

\begin{equation}i_{t}=\sigma\left(W_{x i} * X_{t}+W_{h i} * H_{t-1}+W_{c i} \circ C_{t-1}+b_{i}\right)\end{equation}
\begin{equation}f_{t}=\sigma\left(W_{x f} * X_{t}+W_{h f} * H_{t-1}+W_{c f} \circ C_{t-1}+b_{f}\right)\end{equation}
\begin{equation}C_{t}=f_{t} \circ C_{t-1}+i_{t} \circ \tanh \left(W_{x c} * X_{t}+W_{h c} * H_{t-1}+b_{c}\right)\end{equation} \begin{equation}o_{t}=\sigma\left(W_{x o} * X_{t}+W_{h o} * H_{t-1}+W_{c o} \circ C_{t}+b_{o}\right)\end{equation}
\begin{equation}H_{t}=o_{t} \circ \tanh \left(C_{t}\right)\end{equation}

$\*$ denotes the convolution operation and $\circ$ denotes the Hadamard product like before.  

Can you spot the subtle difference between these equations and regular LSTM? We simply replace the multiplications in the four gates between a) our weight matrices and input ($W_{x} x_{t}$ with $W_{x} * X_{t}$) and b) our weight matrices and previous hidden state ($W_{h} h_{t-1}$ with $W_{h} * H_{t-1}$). Otherwise, everything remains the same.  


If you prefer not to dive into the above equations, the primary thing to note is the fact that we use convolutions (kernel) to process our input images to derive feature maps rather than vectors derived from fully-connected layers. 

### _n_-step Ahead Prediction

The most difficult thing when designing frame prediction models (with ConvLSTM) is defining how to produce the frame predictions. We list two methods here (but others do also exist):

1. Predict the next frame and feed it back into the network for a number of _n_ steps to produce _n_ frame predictions.
2. Predict all future time steps in one-go by having the number of ConvLSTM layers _l_ be equal to the number of _n_ steps. Thus, we can simply use the output from each decoder LSTM cell as our predictions

In this tutorial we will focus on number 1 - especially since it can produce any number of predictions in the future without having to change the architecture completely. Furthermore, if we are to predict many steps in the future option 2 becomes increasingly computationally expensive.

### Lets write some code!

For our ConvLSTM implementation we use the pytorch implementation from [ndrplz](https://raw.githubusercontent.com/ndrplz/ConvLSTM_pytorch/master/convlstm.py)

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

