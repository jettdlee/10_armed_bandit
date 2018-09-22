# 10 Armed Bandit
##### For Assignments COMP532 Machine Learning and BioInspired Optimisation, MSc Computer Science, University of Liverpool 
Python implementation to compare the greedy and e-greedy methods in a 10-armed bandit testbed, presented in [Sutton, Richard S., and Andrew G. Barto. *Reinforcement learning: An introduction*. MIT press, 1998](http://incompleteideas.net/book/bookdraft2017nov5.pdf), Figure 2.2.

Dependencies:
* Numpy
* Matplotlib
```
python 10_bandit_testbed.py
```

Set of 2000 randomly generated n-armed bandit problems with n=10. 
For each problem, action values Q* (a),a=1,…,10, selected according to the normal (Gaussian distribution) with mean = 0 and variance = 1.
Learning method applied to problem selected action A_t, at time step t, actual reward R_t, was selected from a normal distribution with mean Q* (A_t) and variance = 1. 
learning method, we measure its performance and behaviour as it improves with experience over 100 time steps, when applied to one bandit problem, 1 run.
Repeated over 2000 independent runs, each with a different bandit problem, we obtained measures of the learning algorithms behaviour.

Comparison of the greedy with two ε-greedy methods (values = 0.1 & 0.01). All the methods formed their action-value estimates using the sample average technique. 
Upper graph shows the increase in expected reward with experience. Greedy method improved slightly faster, but levelled off at lower levels, achieving a reward of about 1 compared to the max 1.55. The greedy method performed worse in the long run as it got stuck performing sub optimal actions. Lower graph shows greedy method found optimal actions 1/3 of the task. ε-greedy method performed better due to exploration and improve chances of recognizing the optimal action. 
Advantage of greedy over ε depends on the task. With nosier rewards, it requires more exploration to find the optimal, with the ε being better. However, if the reward was zero, greedy would know the true value of each action as it will soon find the optimal and never explore. However, in deterministic cases, there is a large advantage in exploring if we weaken other assumptions. 
