import mdp_algms
import task_structure
import plotter
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import seaborn as sns
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
STATES_NO = 22+1  # one extra state for completing nothing
STATES = np.arange(STATES_NO)

# allow as many units as possible based on state
ACTIONS = [np.arange(STATES_NO-i) for i in range(STATES_NO)]

HORIZON = 16  # no. of weeks for task
DISCOUNT_FACTOR_REWARD = 0.9  # discounting factor
DISCOUNT_FACTOR_COST = 0.5
EFFICACY = 0.8  # self-efficacy (probability of progress for each unit)

# utilities :
REWARD_THR = 1.0  # reward per unit at threshold (14 units)
REWARD_EXTRA = REWARD_THR/4  # reward per unit after threshold upto 22 units
REWARD_SHIRK = 0.1
EFFORT_WORK = -0.3

# %%
# define environment and reward structure

reward_func = task_structure.reward_threshold(
    STATES, ACTIONS, REWARD_SHIRK, REWARD_THR, REWARD_EXTRA)

effort_func = task_structure.effort(STATES, ACTIONS, EFFORT_WORK)

reward_func_last = np.zeros(len(STATES))
effort_func_last = np.zeros(len(STATES))
T = task_structure.T_binomial(STATES, ACTIONS, EFFICACY)

# %%
# plot the policy

V_opt_full, policy_opt_full, Q_values_full = (
    mdp_algms.find_optimal_policy_diff_discount_factors(
        STATES, ACTIONS, HORIZON, DISCOUNT_FACTOR_REWARD,
        DISCOUNT_FACTOR_COST, reward_func, effort_func, reward_func_last,
        effort_func_last, T)
)

policy_init_state = [policy_opt_full[i][0] for i in range(HORIZON)]
policy_init_state = np.array(policy_init_state)
f, ax = plt.subplots(figsize=(5, 4), dpi=300)
cmap = mpl.colormaps['winter']
sns.heatmap(policy_init_state, linewidths=.5, cmap=cmap, cbar=True)
ax.set_xlabel('timestep')
ax.set_ylabel('horizon')
ax.tick_params(axis='x', labelrotation=90)
colorbar = ax.collections[0].colorbar
colorbar.set_label('actions:\n no. of units', rotation=270, labelpad=25)

plt.savefig(
    'plots/vectors/defections_policy.svg',
    format='svg', dpi=300,  bbox_inches='tight'
)

# %%
# solve for policy given task structure
discount_factors_cost = [0.9, 0.7, 0.6, 0.5]

initial_state = 0
beta = 7

colors = ['crimson', 'indigo', 'tab:blue', 'orange']
plt.figure(figsize=(5, 4), dpi=300)

for i_d, discount_factor_cost in enumerate(discount_factors_cost):

    V_opt_full, policy_opt_full, Q_values_full = (
        mdp_algms.find_optimal_policy_diff_discount_factors(
            STATES, ACTIONS, HORIZON, DISCOUNT_FACTOR_REWARD,
            discount_factor_cost, reward_func, effort_func, reward_func_last,
            effort_func_last, T)
    )

    # effective policy followed by agent
    effective_policy = np.array(
        [[policy_opt_full[HORIZON-1-i][i_s][i] for i in range(HORIZON)]
         for i_s in range(len(STATES))]
    )
    effective_Q = []
    for i_s in range(len(STATES)):
        Q_s_temp = []
        for i in range(HORIZON):
            Q_s_temp.append(Q_values_full[HORIZON-1-i][i_s][:, i])
        effective_Q.append(np.array(Q_s_temp).T)

    trajectories = []
    for i in range(1000):

        s, a = mdp_algms.forward_runs_prob(
            softmax_policy, effective_Q, ACTIONS, initial_state,
            HORIZON, STATES, T, beta)
        trajectories.append(s/2)

    plotter.sausage_plots(trajectories, colors[i_d], HORIZON, 0.2)
    plotter.example_trajectories(trajectories, colors[i_d], 1.5, 3)

sns.despine()
plt.xticks([0, 7, 15])
plt.xlabel('time (weeks)')
plt.ylabel('research hours \n completed')

plt.savefig(
    'plots/vectors/defections_traj.svg',
    format='svg', dpi=300,  bbox_inches='tight'
)
