import seaborn as sns
import mdp_algms
import task_structure
import plotter
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
mpl.rcParams['font.size'] = 24
mpl.rcParams['lines.linewidth'] = 3

# %%


def deterministic_policy(a):
    p = np.where(a == np.max(a), 1, 0)
    return p / sum(p)


def softmax_policy(a, beta):
    c = a - np.max(a)
    p = np.exp(beta*c) / np.sum(np.exp(beta*c))
    return p

# %%
# set parameters


# states of markov chain
STATES_NO = 22+1  # 22 one extra state for completing nothing
STATES = np.arange(STATES_NO)

# allow as many units as possible based on state
ACTIONS = [np.arange(STATES_NO-i) for i in range(STATES_NO)]

HORIZON = 15  # no. of weeks for task
DISCOUNT_FACTOR = 0.9  # discounting factor
EFFICACY = 0.8  # self-efficacy (probability of progress for each unit)

# utilities :
REWARD_THR = 4.0  # reward per unit at threshold (14 units)
REWARD_EXTRA = REWARD_THR/8  # reward per unit after threshold upto 22 units
REWARD_SHIRK = 0.1
EFFORT_WORK = -0.3

# %%
# define environment and reward structure
# rewards as soon as 14 credits are hit

reward_func = task_structure.reward_threshold(
    STATES, ACTIONS, REWARD_SHIRK, REWARD_THR, REWARD_EXTRA)

effort_func = task_structure.effort(STATES, ACTIONS, EFFORT_WORK)

total_reward_func_last = np.zeros(len(STATES))

# total reward= reward+effort
total_reward_func = []
for state_current in range(len(STATES)):

    total_reward_func.append(reward_func[state_current]
                             + effort_func[state_current])

T = task_structure.T_binomial(STATES, ACTIONS, EFFICACY)

# %%
# base policy with single discount factor

discount_factors = [1.0, 0.9]

initial_state = 0
beta = 7

colors = ['indigo', 'tab:blue', 'orange']
plt.figure(figsize=(5, 4), dpi=300)

for i_dis, discount_factor in enumerate(discount_factors):

    T = task_structure.T_binomial(STATES, ACTIONS, EFFICACY)

    V_opt, policy_opt, Q_values = mdp_algms.find_optimal_policy_prob_rewards(
        STATES, ACTIONS, HORIZON, discount_factor,
        total_reward_func, total_reward_func_last, T)

    trajectories = []
    for i in range(1000):

        s, a = mdp_algms.forward_runs_prob(
            softmax_policy, Q_values, ACTIONS, initial_state, HORIZON, STATES,
            T, beta)
        trajectories.append(s/2)

    plotter.sausage_plots(trajectories, colors[i_dis], HORIZON, 0.2)
    plotter.example_trajectories(trajectories, colors[i_dis], 1.5, 3)

sns.despine()
plt.xticks([0, 7, 15])
plt.xlabel('time (weeks)')
plt.ylabel('research hours \n completed')

plt.savefig(
    'plots/vectors/no_delay_discounts.svg',
    format='svg', dpi=300,  bbox_inches='tight'
)


# %%
# what if there is a cost related to the number of units

EXPONENT = 2.0  # to make effort function more convex

initial_state = 0
beta = 7

colors = ['indigo', 'tab:blue', 'orange']
plt.figure(figsize=(5, 4), dpi=300)

discount_factor = 0.6
exponents = [1.5, 2.2]

for i_exp, exponent in enumerate(exponents):

    reward_func = task_structure.reward_threshold(
        STATES, ACTIONS, REWARD_SHIRK, REWARD_THR, REWARD_EXTRA)

    effort_func = task_structure.effort_convex_concave(STATES, ACTIONS,
                                                       EFFORT_WORK, exponent)

    total_reward_func_last = np.zeros(len(STATES))

    # total reward = reward+effort
    total_reward_func = []
    for state_current in range(len(STATES)):

        total_reward_func.append(reward_func[state_current]
                                 + effort_func[state_current])

    T = task_structure.T_binomial(STATES, ACTIONS, EFFICACY)

    V_opt, policy_opt, Q_values = mdp_algms.find_optimal_policy_prob_rewards(
        STATES, ACTIONS, HORIZON, discount_factor,
        total_reward_func, total_reward_func_last, T)

    trajectories = []
    for i in range(1000):

        s, a = mdp_algms.forward_runs_prob(
            softmax_policy, Q_values, ACTIONS, initial_state, HORIZON, STATES,
            T, beta)
        trajectories.append(s/2)

    plotter.sausage_plots(trajectories, colors[i_exp], HORIZON, 0.2)
    plotter.example_trajectories(trajectories, colors[i_exp], 1.5, 3)

sns.despine()


# higher discount factor

discount_factor = 0.9
exponent = 2.2

reward_func = task_structure.reward_threshold(
    STATES, ACTIONS, REWARD_SHIRK, REWARD_THR, REWARD_EXTRA)

effort_func = task_structure.effort_convex_concave(STATES, ACTIONS,
                                                   EFFORT_WORK, exponent)

total_reward_func_last = np.zeros(len(STATES))

# total reward = reward+effort
total_reward_func = []
for state_current in range(len(STATES)):

    total_reward_func.append(reward_func[state_current]
                             + effort_func[state_current])

T = task_structure.T_binomial(STATES, ACTIONS, EFFICACY)

V_opt, policy_opt, Q_values = mdp_algms.find_optimal_policy_prob_rewards(
    STATES, ACTIONS, HORIZON, discount_factor,
    total_reward_func, total_reward_func_last, T)

trajectories = []
for i in range(1000):

    s, a = mdp_algms.forward_runs_prob(
        softmax_policy, Q_values, ACTIONS, initial_state, HORIZON, STATES,
        T, beta)
    trajectories.append(s/2)

plotter.sausage_plots(trajectories, colors[i_exp+1], HORIZON, 0.2)
plotter.example_trajectories(trajectories, colors[i_exp+1], 1.5, 3)

sns.despine()
plt.xticks([0, 7, 15])
plt.xlabel('time (weeks)')
plt.ylabel('research hours \n completed')

plt.savefig(
    'plots/vectors/no_delay_convexity.svg',
    format='svg', dpi=300,  bbox_inches='tight'
)
