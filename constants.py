"""
define quantities that are held constant across simulations
"""
import numpy as np

# states of markov chain
STATES_NO = 22+1  # one extra state for completing nothing
STATES = np.arange(STATES_NO)

# actions = no. of units to complete in each state
# allow as many units as possible based on state
ACTIONS = [np.arange(STATES_NO-i) for i in range(STATES_NO)]

HORIZON = 16  # no. of weeks for task

REWARD_SHIRK = 0.1  # rewards for shirk
EFFORT_WORK = -0.3  # costs for work
THR = 14  # threshold number of units for rewards

BETA = 7  # inv temperature for softmax
