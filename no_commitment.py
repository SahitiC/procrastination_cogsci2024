import mdp_algms
import task_structure
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['font.size'] = 14
mpl.rcParams['lines.linewidth'] = 2

# %%


def deterministic_policy(a):
    p = np.where(a == np.max(a), 1, 0)
    return p / sum(p)


def softmax_policy(a, beta):
    c = a - np.max(a)
    p = np.exp(beta*c) / np.sum(np.exp(beta*c))
    return p

# %%


# states of markov chain
STATES_NO = (22+1) * 2  # each state can be high or low rewarding (so, double)
STATES = np.arange(STATES_NO)

# allow as many units as possible based on state
ACTIONS_BASE = [np.arange(int(STATES_NO/2-i)) for i in range(int(STATES_NO/2))]
ACTIONS = ACTIONS_BASE.copy()
# same actions available for low and high reward states: so repeat
ACTIONS.extend(ACTIONS_BASE)

HORIZON = 15  # no. of weeks for task
DISCOUNT_FACTOR = 1.0  # discounting factor
EFFICACY = 0.6  # self-efficacy (probability of progress for each unit)
P_STAY = 0.95  # probability of switching between reward states

# utilities :
REWARD_UNIT = 4.0  # reward per unit at threshold (14 units)
REWARD_EXTRA = 0  # reward per unit after threshold upto 22 units
REWARD_INTEREST = 2.0  # immediate interest rewards on top of course rewards
REWARD_SHIRK = 0.1
EFFORT_WORK = -0.3


# %%
# define reward structure
# define separately for low and high rewards states and then extend

# base course rewards (threshold of 14 units)
reward_func_base = task_structure.reward_threshold(
    STATES[:int(STATES_NO/2)], ACTIONS_BASE, REWARD_SHIRK,
    REWARD_UNIT, REWARD_EXTRA)

# immediate interest rewards
reward_func_interest = task_structure.reward_immediate(
    STATES[:int(STATES_NO/2)], ACTIONS_BASE, 0, REWARD_INTEREST,
    REWARD_INTEREST)

# effort costs
effort_func = task_structure.effort(STATES[:int(STATES_NO/2)], ACTIONS_BASE,
                                    EFFORT_WORK)

# total reward for low reward state = reward_base + effort
total_reward_func_low = []
for state_current in range(len(STATES[:int(STATES_NO/2)])):

    temp = reward_func_base[state_current] + effort_func[state_current]
    # replicate rewards for high reward states
    total_reward_func_low.append(np.block([temp, temp]))

# total reward for high reward state = reward_base + interest rewards + effort
total_reward_func_high = []
for state_current in range(len(STATES[:int(STATES_NO/2)])):

    temp = (reward_func_base[state_current]
            + reward_func_interest[state_current]
            + effort_func[state_current])
    total_reward_func_high.append(np.block([temp, temp]))

total_reward_func = []
total_reward_func.extend(total_reward_func_low)
total_reward_func.extend(total_reward_func_high)

total_reward_func_last = np.zeros(len(STATES))  # final rewards

# %%
# define transition structure
# define separately for low and high rewards states and then extend

T_partial = task_structure.T_binomial(STATES[:int(STATES_NO/2)], ACTIONS_BASE,
                                      EFFICACY)
T_low = []
for state_current in range(len(STATES[:int(STATES_NO/2)])):

    temp = np.block([P_STAY * T_partial[state_current],
                     (1 - P_STAY) * T_partial[state_current]])
    assert (np.round(np.sum(temp, axis=1), 6) == 1).all()
    T_low.append(temp)

T_high = []
for state_current in range(len(STATES[:int(STATES_NO/2)])):

    temp = np.block([(P_STAY) * T_partial[state_current],
                     (1 - P_STAY) * T_partial[state_current]])
    assert (np.round(np.sum(temp, axis=1), 6) == 1).all()
    T_high.append(temp)

T = []
T.extend(T_low)
T.extend(T_high)

# %%
# solve for optimal policy

V_opt, policy_opt, Q_values = mdp_algms.find_optimal_policy_prob_rewards(
    STATES, ACTIONS, HORIZON, DISCOUNT_FACTOR,
    total_reward_func, total_reward_func_last, T)

initial_state = 0
beta = 7
plt.figure()
for i in range(20):
    s, a = mdp_algms.forward_runs_prob(
        softmax_policy, Q_values, ACTIONS, initial_state, HORIZON, STATES,
        T, beta)
    # convert state values to number of units done
    s_unit = np.where(s > 22, s-23, s)
    plt.plot(s_unit, color='gray')
plt.plot(s_unit, color='gray', label='softmax noise')

initial_state = 0
s, a, v = mdp_algms.forward_runs(
    policy_opt, V_opt, initial_state, HORIZON, STATES, T)
plt.plot(s_unit, label='deterministic')
