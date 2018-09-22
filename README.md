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
