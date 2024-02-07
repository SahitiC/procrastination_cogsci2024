import mdp_algms
import task_structure
import constants
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from tslearn.clustering import TimeSeriesKMeans
import seaborn as sns
mpl.rcParams['font.size'] = 24
mpl.rcParams['lines.linewidth'] = 2

# %%

# states of markov chain
STATES_NO = (22+1) * 2  # each state can be high or low rewarding (so, double)
STATES = np.arange(STATES_NO)

# allow as many units as possible based on state
ACTIONS_BASE = [np.arange(int(STATES_NO/2-i)) for i in range(int(STATES_NO/2))]
ACTIONS = ACTIONS_BASE.copy()
# same actions available for low and high reward states: so repeat
ACTIONS.extend(ACTIONS_BASE)

DISCOUNT_FACTOR = 1.0  # discounting factor
EFFICACY = 0.5  # self-efficacy (probability of progress for each unit)
P_STAY = 0.95  # probability of switching between reward states

# utilities :
REWARD_UNIT = 4.0  # reward per unit at threshold (14 units)
REWARD_EXTRA = 0  # reward per unit after threshold upto 22 units
REWARD_INTEREST = 2.0  # immediate interest rewards on top of course rewards


# %%
# solve for optimal policy
# when high rewards stay for long
initial_state = 0

colors = ['indigo', 'tab:blue']
plt.figure(figsize=(5, 4), dpi=300)

reward_interests = [2.0, 0.2]

T = task_structure.get_transitions_interest_states(
    STATES, STATES_NO, ACTIONS_BASE, EFFICACY, P_STAY, P_STAY)

for i_ri, reward_interest in enumerate(reward_interests):

    total_reward_func, total_reward_func_last = task_structure.generate_interest_rewards(
        STATES, STATES_NO, ACTIONS_BASE, constants.REWARD_SHIRK, REWARD_UNIT,
        REWARD_EXTRA, constants.EFFORT_WORK, reward_interest)

    V_opt, policy_opt, Q_values = mdp_algms.find_optimal_policy_prob_rewards(
        STATES, ACTIONS, constants.HORIZON, DISCOUNT_FACTOR,
        total_reward_func, total_reward_func_last, T)

    for i in range(3):
        s, a = mdp_algms.forward_runs_prob(
            task_structure.softmax_policy, Q_values, ACTIONS, initial_state,
            constants.HORIZON, STATES, T, constants.BETA)
        # convert state values to number of units done
        s_unit = np.where(s > constants.STATES_NO-1, s-constants.STATES_NO, s)
        plt.plot(s_unit/2, color=colors[i_ri])


# low discount factor
discount_factor = 0.9
reward_interest = 2.0
total_reward_func, total_reward_func_last = task_structure.generate_interest_rewards(
    STATES, STATES_NO, ACTIONS_BASE, constants.REWARD_SHIRK, REWARD_UNIT,
    REWARD_EXTRA, constants.EFFORT_WORK, reward_interest)


total_reward_func, total_reward_func_last = task_structure.generate_interest_rewards(
    STATES, STATES_NO, ACTIONS_BASE, constants.REWARD_SHIRK, REWARD_UNIT,
    REWARD_EXTRA, constants.EFFORT_WORK, reward_interest)

V_opt, policy_opt, Q_values = mdp_algms.find_optimal_policy_prob_rewards(
    STATES, ACTIONS, constants.HORIZON, discount_factor,
    total_reward_func, total_reward_func_last, T)

for i in range(3):
    s, a = mdp_algms.forward_runs_prob(
        task_structure.softmax_policy, Q_values, ACTIONS, initial_state,
        constants.HORIZON, STATES, T, constants.BETA)
    # convert state values to number of units done
    s_unit = np.where(s > 22, s-23, s)
    plt.plot(s_unit/2, color='orange')

sns.despine()
plt.xticks([0, 7, 15])
plt.xlabel('time (weeks)')
plt.ylabel('research hours \n completed')

plt.savefig(
    'plots/vectors/no_commit_discount.svg',
    format='svg', dpi=300,  bbox_inches='tight'
)

# %%
# for switchy transitions
reward_interest = 2.0
efficacy = 0.6
colors = ['hotpink', 'gray']
plt.figure(figsize=(5, 4), dpi=300)

T = task_structure.get_transitions_interest_states(
    STATES, STATES_NO, ACTIONS_BASE, efficacy, P_STAY, 1-P_STAY)

total_reward_func, total_reward_func_last = task_structure.generate_interest_rewards(
    STATES, STATES_NO, ACTIONS_BASE, constants.REWARD_SHIRK, REWARD_UNIT,
    REWARD_EXTRA, constants.EFFORT_WORK, reward_interest)

V_opt, policy_opt, Q_values = mdp_algms.find_optimal_policy_prob_rewards(
    STATES, ACTIONS, constants.HORIZON, DISCOUNT_FACTOR,
    total_reward_func, total_reward_func_last, T)

series = []
timeseries_to_cluster = []
for i in range(10):
    s, a = mdp_algms.forward_runs_prob(
        task_structure.softmax_policy, Q_values, ACTIONS, initial_state,
        constants.HORIZON, STATES, T, constants.BETA)
    # convert state values to number of units done
    s_unit = np.where(s > constants.STATES_NO-1, s-constants.STATES_NO, s)
    series.append(s_unit/2)
    timeseries_to_cluster.append(s_unit/(s_unit[-1]*2))

# cluster timeseries
km = TimeSeriesKMeans(n_clusters=2, n_init=5, metric="euclidean", verbose=True)
timeseries_to_cluster = np.vstack(timeseries_to_cluster)
labels = km.fit_predict(timeseries_to_cluster)

for i in range(10):
    plt.plot(series[i], color=colors[labels[i]])

sns.despine()
plt.xticks([0, 7, 15])
plt.xlabel('time (weeks)')
plt.ylabel('research hours \n completed')

plt.savefig(
    'plots/vectors/no_commit_switchy.svg',
    format='svg', dpi=300,  bbox_inches='tight'
)
