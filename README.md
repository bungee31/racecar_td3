# Racecar project

The project solves automated navigation in MIT Racecar environment. Uses Deep Learning approach, both DDPG and TD3 implementations.

# INTRODUCTION
Given a simulated environment our task is to have an agent which would reach the goal from a target position by avoiding obstacles. The simulation consists of a 2D environment - a map which can be randomly generated and an agent. We can define the shape of the figure - a simplistic shape would consist of a straight line environment and then we can increase the environment complexity by increasing the number of turns in the map. The target and goal locations are also random (within the boundaries of the map). The agent would have to choose an appropriate speed and steering to navigate through the map.

The objective in our case is to develop a strategy, a deterministic policy that would most accurate calculate the discontinued future reward - to act in a stochastic environment with a continuous action space. 

# DDPG
Given the context of the problem, to solve the task we chose to use the Deep Deterministic Policy Gradient algorithm. This is an off policy method that learns how to approximate a Q-function and a policy. It first uses the Bellman equation to learn the Q function -> which the is used to derive and learn the policy.
It was specifically developed for dealing with continuous action space environments - which makes it perfect for our use case.
Uses the Actor Critic scenario - it being a combination and improvement of value and policy based methods. 
The actor - being the one who performs the action - is policy based - IT USES THE POLICY to map an optimal action given a certain state. 
The critic - as the name implies - is the one which measures the quality of the action selected by the actor - thus GIVEN A STATE AS AND ACTION AS an imput - it should output the reward.
To learn how to better approximate the policy - we use the Double Deep Q Learning. This way we have two independent estimators which can compute unbiased Q value estimates for the actions selected. One estimator takes care of the discounted future reward - given that we have already taken an action. The other estimator returns the predicted Q value for the current state and action. 
These values are sed to compute the loss function. In order for the estimators to be improved with every iteration, the loss function is computed and used to backpropagate the weights. 

# TD3

TD3 works in a very similar way, but the main differences are that the critic consists of two different (but identical) networks. 
Then, instead of backpropagating the weights every time the policy is trained, TD3 uses a delayed policy in which we only update the wrights and the targets with specific frequency. 
Clipped DDQN is used instead of DDQN which was used in DDPG. This means that when selecting an action during training, a clipped Gaussian noise is added to the action to add more randomization. 

# Running the code

The default arguments are solving the problem using TD3. So to use the defaults just perform the following using command line:

python main.py

In order to solve it using DDPG approach run it as follows:

python main.py --policy DDPG





