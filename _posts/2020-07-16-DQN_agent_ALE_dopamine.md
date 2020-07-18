# Deep Reinforcement Learning for Atari Games using Dopamine


In this post, we will look into training a DQN agent for a Atari games using the Google reinforcement learning library [Dopamine](https://github.com/google/dopamine).
While many RL libraries exists, this library is specifically designed with four essential features in mind:
- Easy experimentation
- Flexible development
- Compact and reliable
- Reproducible

We believe these principles makes Dopamine one of the best RL learning environment available today. 
Additionally, we even got the library to work on Windows - which we think is quite a feat!



In my view, visualization of any trained RL agent is an absolutely must in reinforcement learning! 
Therefore, we will (of course) include this at the very end!


## Brief introduction to Reinforcement Learning and Deep Q-Learning



## Installation
This guide does not include instructions for installing [Tensorflow](https://www.tensorflow.org/), but we do want to stress that you can use both CPU and GPU version.
We encountered some issues related to GPU vs. CPU in our setting, so I will include some tips and tricks at the end, which might help you debug your specific installation. 

Nevertheless, assuming you are using ```Python 3.7.x```, these are the libraries you need to install (which can all be installed via ```pip```):

```
tensorflow=1.15
cmake
dopamine-rl
atari-py
matplotlib
pygame
seaborn
pandas
```


## Training our agent 




## Visualizing our agent

Replace the following ```<dopamine/directory>``` and ```<model_path/directory/tf_ckpt-199>``` with your working directory and saved model directory paths, respectively.

```bash
python example_viz.py --agent=rainbow --game=SpaceInvaders --num_steps=1000 --root_dir=<dopamine/directory> --restore_checkpoint=<model_path/directory>/tf_ckpt-199
```


## Tips and tricks (debugging)

You might run into the same issue I did, which was related to my tensorflow installation. 
Even though I installed the CPU version, the tensorflow libraries looks for my CUDA device when doing computations. 
If you already have the tensorflow GPU version installed and working then no worries - but for this post, I have decided to use the CPU version so everyone can follow along.

Nevertheless, add the following two lines to your script before doing any tensorflow computations!

```python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
```
