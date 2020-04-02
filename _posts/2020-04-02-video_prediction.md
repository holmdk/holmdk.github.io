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

For our ConvLSTM implementation we use the implementation from the [CortexNet](https://arxiv.org/pdf/1706.02735.pdf) [Atcold/pytorch-CortexNet](https://github.com/Atcold/pytorch-CortexNet/blob/master/model/ConvLSTMCell.py)

It looks as follows:

```python
class ConvLSTMCell(nn.Module):
    """
    Generate a convolutional LSTM cell
    """

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.Gates = nn.Conv2d(input_size + hidden_size, 4 * hidden_size, KERNEL_SIZE, padding=PADDING)

    def forward(self, input_, prev_state):

        # get batch and spatial sizes
        batch_size = input_.data.size()[0]
        spatial_size = input_.data.size()[2:]

        # generate empty prev_state, if None is provided
        if prev_state is None:
            state_size = [batch_size, self.hidden_size] + list(spatial_size)
            prev_state = (
                Variable(torch.zeros(state_size)),
                Variable(torch.zeros(state_size))
            )

        prev_hidden, prev_cell = prev_state

        # data size is [batch, channel, height, width]
        stacked_inputs = torch.cat((input_, prev_hidden), 1)
        gates = self.Gates(stacked_inputs)

        # chunk across channel dimension
        in_gate, remember_gate, out_gate, cell_gate = gates.chunk(4, 1)

        # apply sigmoid non linearity
        in_gate = f.sigmoid(in_gate)
        remember_gate = f.sigmoid(remember_gate)
        out_gate = f.sigmoid(out_gate)

        # apply tanh non linearity
        cell_gate = f.tanh(cell_gate)

        # compute current cell and hidden state
        cell = (remember_gate * prev_cell) + (in_gate * cell_gate)
        hidden = out_gate * f.tanh(cell)

        return hidden, cell


```

