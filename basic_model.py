"""
code for reproducing Figure 2 A-D;
what it does: define MDP for exponential discounting with delayed rewards,
simulate for different configurations of parameters
"""
import mdp_algms
import task_structure
import plotter
import constants
import compute_distance
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import pandas as pd
import numpy as np
import ast
mpl.rcParams['font.size'] = 24
mpl.rcParams['lines.linewidth'] = 3

SAVE_ONLY = True

# %%
# import real data (clustered)
data = pd.read_csv('data_relevant_clustered.csv')
literal_array = data['cumulative_progress_weeks'].apply(ast.literal_eval)
cumulative_progress_weeks = np.array(literal_array.tolist())
labels = np.array(data['labels'])

# %%
# instantiate MDP

DISCOUNT_FACTOR = 0.9  # discounting factor
EFFICACY = 0.8  # self-efficacy (probability of progress for each unit)

# utilities :
REWARD_THR = 4.0  # reward per unit at threshold (14 units)
REWARD_EXTRA = REWARD_THR/4  # reward per unit after threshold upto 22 units

# %%
# reward structure: with rewards for units only at deadline but immediate
# shirk rewards and effort costs

reward_func = task_structure.reward_no_immediate(
    constants.STATES, constants.ACTIONS, constants.REWARD_SHIRK)

effort_func = task_structure.effort(
    constants.STATES, constants.ACTIONS, constants.EFFORT_WORK)

total_reward_func_last = task_structure.reward_final(
    constants.STATES, REWARD_THR, REWARD_EXTRA, constants.THR,
    constants.STATES_NO)

# total reward= reward+effort
total_reward_func = []
for state_current in range(len(constants.STATES)):

    total_reward_func.append(reward_func[state_current]
                             + effort_func[state_current])

# %%
# mean trajectories (with std.) across discount and efficacy

discount_factor = 1.0
efficacies = [0.98, 0.6, 0.3]

colors = ['indigo', 'tab:blue', 'orange']
plt.figure(figsize=(5, 4), dpi=300)

for i_efficacy, efficacy in enumerate(efficacies):

    # define transition probabilities
    T = task_structure.T_binomial(
        constants.STATES, constants.ACTIONS, efficacy)

    # calculate optimal policy
    V_opt, policy_opt, Q_values = mdp_algms.find_optimal_policy_prob_rewards(
        constants.STATES, constants.ACTIONS, constants.HORIZON,
        discount_factor, total_reward_func, total_reward_func_last, T)

    trajectories = []
    for _ in range(constants.N_TRIALS):

        s, a = mdp_algms.forward_runs_prob(
            task_structure.softmax_policy, Q_values, constants.ACTIONS,
            constants.INITIAL_STATE, constants.HORIZON, constants.STATES, T,
            constants.BETA)
        trajectories.append(s)

    plotter.sausage_plots(
        trajectories, colors[i_efficacy], constants.HORIZON, 0.2)
    plotter.example_trajectories(trajectories, colors[i_efficacy], 1.5, 3)

    # compare simulated trajectories to data clusters by calculating avg
    # distance to trajectories of each cluster
    # ignore first entry of simulated trajectory (as it is always 0)
    distance_no_discount = compute_distance.avg_distance_all_clusters(
        cumulative_progress_weeks, labels, np.array(trajectories)[:, 1:])

sns.despine()
plt.xticks([0, 7, 15])
# add tick at threshold (14 units):
plt.yticks(list(plt.yticks()[0][1:-1]) + [constants.THR])
plt.xlabel('time (weeks)')
plt.ylabel('research units \n completed')

plt.savefig(
    'plots/vectors/basic_no_discount.svg',
    format='svg', dpi=300)


# %%
# mean trajectories (with std.) across discount and efficacy

discount_factor = 0.9
efficacies = [0.98, 0.6, 0.3]

colors = ['indigo', 'tab:blue', 'orange']
plt.figure(figsize=(5, 4), dpi=300)

for i_efficacy, efficacy in enumerate(efficacies):

    # define transition probabilities
    T = task_structure.T_binomial(
        constants.STATES, constants.ACTIONS, efficacy)

    # calculate optimal policy
    V_opt, policy_opt, Q_values = mdp_algms.find_optimal_policy_prob_rewards(
        constants.STATES, constants.ACTIONS, constants.HORIZON,
        discount_factor, total_reward_func, total_reward_func_last, T)

    trajectories = []
    for _ in range(constants.N_TRIALS):

        s, a = mdp_algms.forward_runs_prob(
            task_structure.softmax_policy, Q_values, constants.ACTIONS,
            constants.INITIAL_STATE, constants.HORIZON, constants.STATES, T,
            constants.BETA)
        trajectories.append(s)

    plotter.sausage_plots(
        trajectories, colors[i_efficacy], constants.HORIZON, 0.2)
    plotter.example_trajectories(trajectories, colors[i_efficacy], 1.5, 3)

    # compare simulated trajectories to data clusters by calculating avg
    # distance to trajectories of each cluster
    # ignore first entry of simulated trajectory (as it is always 0)
    distance_discount = compute_distance.avg_distance_all_clusters(
        cumulative_progress_weeks, labels, np.array(trajectories)[:, 1:])

sns.despine()
plt.xticks([0, 7, 15])
# add tick at threshold (14 units):
plt.yticks(list(plt.yticks()[0][1:-1]) + [constants.THR])
plt.xlabel('time (weeks)')
plt.ylabel('research units \n completed')

plt.savefig(
    'plots/vectors/basic_discount.svg',
    format='svg', dpi=300)


# %%
# gap between real and assumed efficacies

assumed_efficacies = [0.6, 0.3, 0.2]
real_efficacy = 0.8

colors = ['hotpink', 'mediumturquoise', 'goldenrod']
plt.figure(figsize=(5, 4), dpi=300)

for i_efficacy, ass_efficacy in enumerate(assumed_efficacies):

    T = task_structure.T_binomial(
        constants.STATES, constants.ACTIONS, ass_efficacy)

    V_opt, policy_opt, Q_values = mdp_algms.find_optimal_policy_prob_rewards(
        constants.STATES, constants.ACTIONS, constants.HORIZON,
        DISCOUNT_FACTOR, total_reward_func, total_reward_func_last, T)

    T_real = task_structure.T_binomial(
        constants.STATES, constants.ACTIONS, real_efficacy)

    trajectories = []
    for _ in range(constants.N_TRIALS):

        s, a = mdp_algms.forward_runs_prob(
            task_structure.softmax_policy, Q_values, constants.ACTIONS,
            constants.INITIAL_STATE, constants.HORIZON, constants.STATES,
            T_real, constants.BETA)
        trajectories.append(s)

    plotter.sausage_plots(
        trajectories, colors[i_efficacy], constants.HORIZON, 0.3)
    plotter.example_trajectories(trajectories, colors[i_efficacy], 1.5, 3)

    distance_eff_gap = compute_distance.avg_distance_all_clusters(
        cumulative_progress_weeks, labels, np.array(trajectories)[:, 1:])

sns.despine()
plt.xticks([0, 7, 15])
plt.yticks(list(plt.yticks()[0][1:-1]) + [constants.THR])
plt.xlabel('time (weeks)')
plt.ylabel('research units \n completed')

plt.savefig(
    'plots/vectors/basic_gap_efficacys.svg',
    format='svg', dpi=300)


# %%
# with convexity
discount_factor = 1.0
convexitys = [1.1, 1.8]

colors = ['indigo', 'tab:blue', 'orange']
plt.figure(figsize=(5, 4), dpi=300)

T = task_structure.T_binomial(constants.STATES, constants.ACTIONS, EFFICACY)

for i_c, convexity in enumerate(convexitys):

    effort_func = task_structure.effort_convex_concave(
        constants.STATES, constants.ACTIONS, constants.EFFORT_WORK, convexity)

    total_reward_func = []
    for state_current in range(len(constants.STATES)):

        total_reward_func.append(reward_func[state_current]
                                 + effort_func[state_current])

    V_opt, policy_opt, Q_values = mdp_algms.find_optimal_policy_prob_rewards(
        constants.STATES, constants.ACTIONS, constants.HORIZON,
        discount_factor, total_reward_func, total_reward_func_last, T)

    trajectories = []
    for _ in range(constants.N_TRIALS):

        s, a = mdp_algms.forward_runs_prob(
            task_structure.softmax_policy, Q_values, constants.ACTIONS,
            constants.INITIAL_STATE, constants.HORIZON, constants.STATES, T,
            constants.BETA)
        trajectories.append(s)

    plotter.sausage_plots(trajectories, colors[i_c], constants.HORIZON, 0.2)
    plotter.example_trajectories(trajectories, colors[i_c], 1.5, 3)

    distance_conv = compute_distance.avg_distance_all_clusters(
        cumulative_progress_weeks, labels, np.array(trajectories)[:, 1:])

sns.despine()
plt.xticks([0, 7, 15])
plt.yticks(list(plt.yticks()[0][1:-1]) + [constants.THR])
plt.xlabel('time (weeks)')
plt.ylabel('research units \n completed')

plt.savefig(
    'plots/vectors/basic_discount_conv.svg',
    format='svg', dpi=300)

if not SAVE_ONLY:
    plt.show()
