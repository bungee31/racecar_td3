# Racecar project

The project solves automated navigation in MIT Racecar environment. Uses Deep Learning approach, both DDPG and TD3 implementations.

## Introduction
Given a simulated environment our task is to have an agent which would reach the goal from a target position by avoiding obstacles. The simulation consists of a 2D environment - a map which can be randomly generated and an agent.

The objective in our case is to develop a strategy, a deterministic policy that would most accurate calculate the discontinued future reward - to act in a stochastic environment with a continuous action space. 

## DDPG
Given the context of the problem, to solve the task we chose to use the Deep Deterministic Policy Gradient algorithm. This is an off policy method that learns how to approximate a Q-function and a policy. It first uses the Bellman equation to learn the Q function -> which the is used to derive and learn the policy.
It was specifically developed for dealing with continuous action space environments - which makes it perfect for our use case.

## TD3

TD3 works in a very similar way, but the main differences are that the critic consists of two different (but identical) networks. 
Then, instead of backpropagating the weights every time the policy is trained, TD3 uses a delayed policy in which we only update the wrights and the targets with specific frequency.

## Running the code

The default arguments are solving the problem using TD3. So to use the defaults just perform the following using command line:

```python
python main.py
```

In order to solve it using DDPG approach run it as follows:

```python
python main.py --policy DDPG
```

The hyperparameters can be changes using the arguments listed in main.py file.




