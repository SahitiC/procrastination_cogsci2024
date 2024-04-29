import mdp_algms
import task_structure
import plotter
import constants
import compute_distance
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import seaborn as sns
import pandas as pd
mpl.rcParams['font.size'] = 24
mpl.rcParams['lines.linewidth'] = 3

# %%
# import real data (clustered)
data = pd.read_csv('data_relevant_clustered.csv')
cumulative_progress_weeks = compute_distance.literal_eval(
    data, 'cumulative_progress_weeks')
labels = np.array(data['labels'])

# %%
# set parameters

DISCOUNT_FACTOR = 0.9  # discounting factor
EFFICACY = 0.8  # self-efficacy (probability of progress for each unit)

# utilities :
REWARD_THR = 4.0  # reward per unit at threshold (14 units)
REWARD_EXTRA = REWARD_THR/8  # reward per unit after threshold upto 22 units


# %%
# define environment and reward structure
# rewards as soon as 14 credits are hit

reward_func = task_structure.reward_threshold(
    constants.STATES, constants.ACTIONS, constants.REWARD_SHIRK, REWARD_THR,
    REWARD_EXTRA, constants.THR, constants.STATES_NO)

effort_func = task_structure.effort(
    constants.STATES, constants.ACTIONS, constants.EFFORT_WORK)

total_reward_func_last = np.zeros(len(constants.STATES))

# total reward= reward+effort
total_reward_func = []
for state_current in range(len(constants.STATES)):

    total_reward_func.append(reward_func[state_current]
                             + effort_func[state_current])

T = task_structure.T_binomial(constants.STATES, constants.ACTIONS, EFFICACY)

# %%
# base policy with single discount factor

discount_factors = [1.0, 0.9]

initial_state = 0

colors = ['indigo', 'tab:blue', 'orange']
plt.figure(figsize=(5, 4), dpi=300)

for i_dis, discount_factor in enumerate(discount_factors):

    T = task_structure.T_binomial(
        constants.STATES, constants.ACTIONS, EFFICACY)

    V_opt, policy_opt, Q_values = mdp_algms.find_optimal_policy_prob_rewards(
        constants.STATES, constants.ACTIONS, constants.HORIZON,
        discount_factor, total_reward_func, total_reward_func_last, T)

    trajectories = []
    for i in range(1000):

        s, a = mdp_algms.forward_runs_prob(
            task_structure.softmax_policy, Q_values, constants.ACTIONS,
            initial_state, constants.HORIZON, constants.STATES, T,
            constants.BETA)
        trajectories.append(s/2)

    plotter.sausage_plots(trajectories, colors[i_dis], constants.HORIZON, 0.2)
    plotter.example_trajectories(trajectories, colors[i_dis], 1.5, 3)

    # compare data clusters to simulated trajectories
    # ignore first entry of simulated trajectory (as it is always 0)
    print(compute_distance.avg_distance_all_clusters(
        cumulative_progress_weeks, labels, np.array(trajectories)[:, 1:]))

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

colors = ['indigo', 'tab:blue', 'orange']
plt.figure(figsize=(5, 4), dpi=300)

discount_factor = 0.6
exponents = [1.5, 2.2]

for i_exp, exponent in enumerate(exponents):

    reward_func = task_structure.reward_threshold(
        constants.STATES, constants.ACTIONS, constants.REWARD_SHIRK,
        REWARD_THR, REWARD_EXTRA, constants.THR, constants.STATES_NO)

    effort_func = task_structure.effort_convex_concave(
        constants.STATES, constants.ACTIONS, constants.EFFORT_WORK, exponent)

    total_reward_func_last = np.zeros(len(constants.STATES))

    # total reward = reward+effort
    total_reward_func = []
    for state_current in range(len(constants.STATES)):

        total_reward_func.append(reward_func[state_current]
                                 + effort_func[state_current])

    T = task_structure.T_binomial(
        constants.STATES, constants.ACTIONS, EFFICACY)

    V_opt, policy_opt, Q_values = mdp_algms.find_optimal_policy_prob_rewards(
        constants.STATES, constants.ACTIONS, constants.HORIZON,
        discount_factor, total_reward_func, total_reward_func_last, T)

    trajectories = []
    for i in range(1000):

        s, a = mdp_algms.forward_runs_prob(
            task_structure.softmax_policy, Q_values, constants.ACTIONS,
            initial_state, constants.HORIZON, constants.STATES, T,
            constants.BETA)
        trajectories.append(s/2)

    plotter.sausage_plots(trajectories, colors[i_exp], constants.HORIZON, 0.2)
    plotter.example_trajectories(trajectories, colors[i_exp], 1.5, 3)

    print(compute_distance.avg_distance_all_clusters(
        cumulative_progress_weeks, labels, np.array(trajectories)[:, 1:]))

sns.despine()


# higher discount factor

discount_factor = 0.9
exponent = 2.2

reward_func = task_structure.reward_threshold(
    constants.STATES, constants.ACTIONS, constants.REWARD_SHIRK, REWARD_THR,
    REWARD_EXTRA, constants.THR, constants.STATES_NO)

effort_func = task_structure.effort_convex_concave(
    constants.STATES, constants.ACTIONS, constants.EFFORT_WORK, exponent)

total_reward_func_last = np.zeros(len(constants.STATES))

# total reward = reward+effort
total_reward_func = []
for state_current in range(len(constants.STATES)):

    total_reward_func.append(reward_func[state_current]
                             + effort_func[state_current])

T = task_structure.T_binomial(constants.STATES, constants.ACTIONS, EFFICACY)

V_opt, policy_opt, Q_values = mdp_algms.find_optimal_policy_prob_rewards(
    constants.STATES, constants.ACTIONS, constants.HORIZON, discount_factor,
    total_reward_func, total_reward_func_last, T)

trajectories = []
for i in range(1000):

    s, a = mdp_algms.forward_runs_prob(
        task_structure.softmax_policy, Q_values, constants.ACTIONS,
        initial_state, constants.HORIZON, constants.STATES, T, constants.BETA)
    trajectories.append(s/2)

plotter.sausage_plots(trajectories, colors[i_exp+1], constants.HORIZON, 0.2)
plotter.example_trajectories(trajectories, colors[i_exp+1], 1.5, 3)

print(compute_distance.avg_distance_all_clusters(
    cumulative_progress_weeks, labels, np.array(trajectories)[:, 1:]))

sns.despine()
plt.xticks([0, 7, 15])
plt.xlabel('time (weeks)')
plt.ylabel('research hours \n completed')

plt.savefig(
    'plots/vectors/no_delay_convexity.svg',
    format='svg', dpi=300,  bbox_inches='tight'
)
