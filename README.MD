# Actor Critic Methods in Reinforcement Learning

<section><p>
As you know there are many approaches about reinforcement 
learning. This repository aims 'actor critic' methods and
'policy optimization techniques'. In first commit, the Proximal Policy
Optimization gradient method was applied to agent with actor critic method.
</p>

<p>
Without PPO (Proximal Policy Optimization) agent also be uploaded. Continuous 
Lunar Landing task is a bit complex, so application of PPO made this task easier. 
</p>
</section>

### What happens without PPO?
<section><p>
This update will be uploaded soon, but it requires widely more epochs than 
PPO applied method. I'm working on it.
</p>
</section>


# About Installation and requirements
```
torch == 1.7.0
torchviz = 0.0.2
gym == 0.25.2
matplotlib == 3.5.3
numpy == 1.23.2
```
Some packages may have been forgotten, if you face errors, feel free to create issue.

# Deep Convolutional Q Learning with PPO
<p align="center">
<img src="https://raw.githubusercontent.com/emredo/reinforcement_learning/master/car_racing_ppo/car_racing.gif">    
</p>

<p align="center">
After 3000 epoch training
</p>

### About experiment

- Episode is terminated when total reward is 800.
- 3000 epoch was enough in my training.

<p align="center">
<img src="https://raw.githubusercontent.com/emredo/reinforcement_learning/master/car_racing_ppo/reward_by_epoch_1663117052.png" alt="Reward by epoch during training">
</p>

<p align="center">
Reward by epoch during training
</p>



### About Backbone
Actor-Critic Network has 1 input head. This input head includes 3 Convolution layer with 12,12,12 filter number, also 2 times max pooling was applied.
After all convolution processes, the flatted matrix is sent 3 different out heads. These heads are *mean, variance* and *value* heads.
The mean head predicts n mean value between -1 and 1 (tanh activation). Variance predicts n variance value between 0 and inf. (softplus activation).
```n is number of actions.```

The last out head is *value* head as we mentioned earlier. This head can be named also *critic network* which predicts value of being state of a state.(V(s))
It hasn't got any activation because V(s) hasn't got exact behaviour.

The layers of 3 head is same, only activations are different. This fcl includes 2 hidden layer with 128 dense. 

### Taking action:
1. The state inputs the network
2. Mean network produces 1 mean value per action in space ```n is number of actions```

    ```mu = self.mu_net(conv_out)``` 
3. Variance network produces 1 variance value per action in space

    ```variance = self.var_net(conv_out)```
4. These means and sqrt(variances) peers per action is used for creating *normal* distribution.
    
    ```dists = t.distributions.Normal(mu, variance ** (1 / 2))```

**Note: When creating distribution, the standard deviation is used, not the variance.**

5. Taking sample from every distribution. (n sample)
    
    ```action = dist.sample()```


# Lunar Landing Continuous with PPO
<p align="center">
<img src="https://raw.githubusercontent.com/emredo/reinforcement_learning/master/lunar_landing_continuous_ppo/lunar_landing.gif" alt="After 3000 epoch training" width="320">    
</p>

Detailed explanation about lunar landing project will be added soon.


### **Note: Any offers or pull requests about repository will be evaluated.**