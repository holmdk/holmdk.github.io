# Deep Q-Learning for Atari Games using Dopamine and the Arcade Learning Environment


In this post, we will look into how we can train a DQN agent to optimize a few Atari games using as little code as possible.

We will also include visualization of the trained agent once we have optimized the RL model.

Let's get started!


## Installation
First things first, lets install the neccessary libraries!



## Tips and tricks

You might run into the same issue I did, which was related to my tensorflow installation. 
Even though I installed the CPU version, the tensorflow libraries looks for my CUDA device when doing computations. 
If you already have the tensorflow GPU version installed and working then no worries - but for this post, I have decided to use the CPU version so everyone can follow along.

Nevertheless, add the following two lines to your script before doing any tensorflow computations!

```python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
```
