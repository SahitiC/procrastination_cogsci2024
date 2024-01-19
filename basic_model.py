import seaborn as sns
import mdp_algms
import task_structure
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
mpl.rcParams['font.size'] = 16
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
# instantiate MDP

# states of markov chain
STATES_NO = 22+1  # one extra state for completing nothing
STATES = np.arange(STATES_NO)

# actions = no. of units to complete in each state
# allow as many units as possible based on state
ACTIONS = [np.arange(STATES_NO-i) for i in range(STATES_NO)]

HORIZON = 16  # no. of weeks for task
DISCOUNT_FACTOR = 0.9  # discounting factor
EFFICACY = 0.7  # self-efficacy (probability of progress for each unit)

# utilities :
REWARD_THR = 4.0  # reward per unit at threshold (14 units)
REWARD_EXTRA = REWARD_THR/4  # reward per unit after threshold upto 22 units
REWARD_SHIRK = 0.1
EFFORT_WORK = -0.3

# %%
# reward structure

reward_func = task_structure.reward_no_immediate(STATES, ACTIONS, REWARD_SHIRK)

effort_func = task_structure.effort(STATES, ACTIONS, EFFORT_WORK)

total_reward_func_last = task_structure.reward_final(STATES, REWARD_THR,
                                                     REWARD_EXTRA)

# total reward= reward+effort
total_reward_func = []
for state_current in range(len(STATES)):

    total_reward_func.append(reward_func[state_current]
                             + effort_func[state_current])


# %%
# variation with discount and efficacy

discount_factor = 0.9
efficacies = [1.0, 0.6, 0.3]

initial_state = 0
beta = 7

colors = ['indigo', 'tab:blue', 'orange']
plt.figure(figsize=(5, 4), dpi=100)

for i_efficacy, efficacy in enumerate(efficacies):

    T = task_structure.T_binomial(STATES, ACTIONS, efficacy)

    V_opt, policy_opt, Q_values = mdp_algms.find_optimal_policy_prob_rewards(
        STATES, ACTIONS, HORIZON, discount_factor,
        total_reward_func, total_reward_func_last, T)

    for i in range(5):

        s, a = mdp_algms.forward_runs_prob(
            softmax_policy, Q_values, ACTIONS, initial_state, HORIZON, STATES,
            T, beta)
        plt.plot(s/2, color=colors[i_efficacy])

    sns.despine()

plt.xlabel('time (weeks)')
plt.ylabel('research hours completed')

# %%
# gap between real and assumed efficacies

assumed_efficacies = [0.7, 0.3, 0.2]
real_efficacy = 1.0

initial_state = 0
beta = 7

colors = ['indigo', 'tab:blue', 'orange']
plt.figure(figsize=(5, 4), dpi=100)

for i_efficacy, ass_efficacy in enumerate(assumed_efficacies):

    T = task_structure.T_binomial(STATES, ACTIONS, ass_efficacy)

    V_opt, policy_opt, Q_values = mdp_algms.find_optimal_policy_prob_rewards(
        STATES, ACTIONS, HORIZON, DISCOUNT_FACTOR,
        total_reward_func, total_reward_func_last, T)

    T_real = task_structure.T_binomial(STATES, ACTIONS, real_efficacy)

    for i in range(5):

        s, a = mdp_algms.forward_runs_prob(
            softmax_policy, Q_values, ACTIONS, initial_state, HORIZON, STATES,
            T_real, beta)
        plt.plot(s/2, color=colors[i_efficacy])

        sns.despine()

    plt.xlabel('time (weeks)')
    plt.ylabel('research hours completed')

# %%
# decreasing efficacy

initial_state = 0
beta = 10
efficacies = [1.0, 0.5]
discount_factor = 1

colors = ['indigo', 'tab:blue', 'orange']
plt.figure(figsize=(5, 4), dpi=100)

for i_efficacy, efficacy in enumerate(efficacies):

    T = task_structure.T_binomial_decreasing(STATES, ACTIONS,
                                             HORIZON, efficacy)

    V_opt, policy_opt, Q_values = mdp_algms.find_optimal_policy_T_time_dep(
        STATES, ACTIONS, HORIZON, DISCOUNT_FACTOR,
        total_reward_func, total_reward_func_last, T)

    for i in range(5):

        s, a = mdp_algms.forward_runs_T_time_dep(
            softmax_policy, Q_values, ACTIONS, initial_state, HORIZON, STATES,
            T, beta)
        plt.plot(s/2, color=colors[i_efficacy])

    sns.despine()

plt.xlabel('time (weeks)')
plt.ylabel('research hours completed')


# %%
# with limits on max number of actions

initial_state = 0
beta = 7
max_units = [10, 4]
discount_factor = 1

colors = ['indigo', 'tab:blue', 'orange']
plt.figure(figsize=(5, 4), dpi=100)


for i_mx, max_unit in enumerate((max_units)):

    actions_lim = []

    for state_current in range(STATES_NO):

        if state_current + max_unit <= STATES_NO-1:
            units = max_unit
        else:
            units = STATES_NO-1-state_current

        actions_lim.append(np.arange(units+1))

    reward_func = task_structure.reward_no_immediate(STATES, actions_lim,
                                                     REWARD_SHIRK)

    effort_func = task_structure.effort(STATES, actions_lim, EFFORT_WORK)

    total_reward_func_last = task_structure.reward_final(STATES, REWARD_THR,
                                                         REWARD_EXTRA)

    # total reward= reward+effort
    total_reward_func = []
    for state_current in range(len(STATES)):

        total_reward_func.append(reward_func[state_current]
                                 + effort_func[state_current])

    T = task_structure.T_binomial(STATES, actions_lim, EFFICACY)

    V_opt, policy_opt, Q_values = mdp_algms.find_optimal_policy_prob_rewards(
        STATES, actions_lim, HORIZON, discount_factor,
        total_reward_func, total_reward_func_last, T)

    for i in range(5):
        s, a = mdp_algms.forward_runs_prob(
            softmax_policy, Q_values, actions_lim, initial_state, HORIZON,
            STATES, T, beta)
        plt.plot(s/2, color=colors[i_mx])

    sns.despine()

plt.xlabel('time (weeks)')
plt.ylabel('research hours completed')
