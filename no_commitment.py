from tslearn.clustering import TimeSeriesKMeans
import seaborn as sns
import mdp_algms
import task_structure
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['font.size'] = 24
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
EFFICACY = 0.5  # self-efficacy (probability of progress for each unit)
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
def generate_interest_rewards(states, states_no, actions_base,
                              reward_shirk, reward_unit, reward_extra,
                              effort_work, reward_interest):
    # base course rewards (threshold of 14 units)
    reward_func_base = task_structure.reward_threshold(
        states[:int(states_no/2)], actions_base, reward_shirk,
        reward_unit, reward_extra)

    # immediate interest rewards
    reward_func_interest = task_structure.reward_immediate(
        states[:int(states_no/2)], actions_base, 0, reward_interest,
        reward_interest)

    # effort costs
    effort_func = task_structure.effort(states[:int(states_no/2)],
                                        actions_base, effort_work)

    # total reward for low reward state = reward_base + effort
    total_reward_func_low = []
    for state_current in range(len(states[:int(states_no/2)])):

        temp = reward_func_base[state_current] + effort_func[state_current]
        # replicate rewards for high reward states
        total_reward_func_low.append(np.block([temp, temp]))

    # total reward for high reward state = base + interest rewards + effort
    total_reward_func_high = []
    for state_current in range(len(states[:int(states_no/2)])):

        temp = (reward_func_base[state_current]
                + reward_func_interest[state_current]
                + effort_func[state_current])
        total_reward_func_high.append(np.block([temp, temp]))

    total_reward_func = []
    total_reward_func.extend(total_reward_func_low)
    total_reward_func.extend(total_reward_func_high)

    total_reward_func_last = np.zeros(len(states))  # final rewards

    return total_reward_func, total_reward_func_last

# %%
# define transition structure
# define separately for low and high rewards states and then extend


def get_transitions_interest_states(states, states_no, actions_base, efficacy,
                                    p_stay_l, p_stay_h):

    T_partial = task_structure.T_binomial(states[:int(states_no/2)],
                                          actions_base, efficacy)
    T_low = []
    for state_current in range(len(states[:int(states_no/2)])):

        temp = np.block([p_stay_l * T_partial[state_current],
                         (1 - p_stay_l) * T_partial[state_current]])
        assert (np.round(np.sum(temp, axis=1), 6) == 1).all()
        T_low.append(temp)

    T_high = []
    for state_current in range(len(STATES[:int(STATES_NO/2)])):

        temp = np.block([(1 - p_stay_h) * T_partial[state_current],
                         (p_stay_h) * T_partial[state_current]])
        assert (np.round(np.sum(temp, axis=1), 6) == 1).all()
        T_high.append(temp)

    T = []
    T.extend(T_low)
    T.extend(T_high)

    return T


# %%
# solve for optimal policy
# when high rewards stay for long
initial_state = 0
beta = 7

colors = ['indigo', 'tab:blue']
plt.figure(figsize=(5, 4), dpi=300)

reward_interests = [2.0, 0.2]

T = get_transitions_interest_states(STATES, STATES_NO, ACTIONS_BASE,
                                    EFFICACY, P_STAY, P_STAY)

for i_ri, reward_interest in enumerate(reward_interests):

    total_reward_func, total_reward_func_last = generate_interest_rewards(
        STATES, STATES_NO, ACTIONS_BASE, REWARD_SHIRK, REWARD_UNIT,
        REWARD_EXTRA, EFFORT_WORK, reward_interest)

    V_opt, policy_opt, Q_values = mdp_algms.find_optimal_policy_prob_rewards(
        STATES, ACTIONS, HORIZON, DISCOUNT_FACTOR,
        total_reward_func, total_reward_func_last, T)

    for i in range(3):
        s, a = mdp_algms.forward_runs_prob(
            softmax_policy, Q_values, ACTIONS, initial_state, HORIZON, STATES,
            T, beta)
        # convert state values to number of units done
        s_unit = np.where(s > 22, s-23, s)
        plt.plot(s_unit/2, color=colors[i_ri])


# low discount factor
discount_factor = 0.9
reward_interest = 2.0
total_reward_func, total_reward_func_last = generate_interest_rewards(
    STATES, STATES_NO, ACTIONS_BASE, REWARD_SHIRK, REWARD_UNIT,
    REWARD_EXTRA, EFFORT_WORK, reward_interest)


total_reward_func, total_reward_func_last = generate_interest_rewards(
    STATES, STATES_NO, ACTIONS_BASE, REWARD_SHIRK, REWARD_UNIT,
    REWARD_EXTRA, EFFORT_WORK, reward_interest)

V_opt, policy_opt, Q_values = mdp_algms.find_optimal_policy_prob_rewards(
    STATES, ACTIONS, HORIZON, discount_factor,
    total_reward_func, total_reward_func_last, T)

for i in range(3):
    s, a = mdp_algms.forward_runs_prob(
        softmax_policy, Q_values, ACTIONS, initial_state, HORIZON, STATES,
        T, beta)
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
colors = ['hotpink', 'mediumturquoise', 'goldenrod']
plt.figure(figsize=(5, 4), dpi=300)

T = get_transitions_interest_states(STATES, STATES_NO, ACTIONS_BASE,
                                    efficacy, P_STAY, 1-P_STAY)


total_reward_func, total_reward_func_last = generate_interest_rewards(
    STATES, STATES_NO, ACTIONS_BASE, REWARD_SHIRK, REWARD_UNIT,
    REWARD_EXTRA, EFFORT_WORK, reward_interest)

V_opt, policy_opt, Q_values = mdp_algms.find_optimal_policy_prob_rewards(
    STATES, ACTIONS, HORIZON, DISCOUNT_FACTOR,
    total_reward_func, total_reward_func_last, T)

series = []
timeseries_to_cluster = []
for i in range(10):
    s, a = mdp_algms.forward_runs_prob(
        softmax_policy, Q_values, ACTIONS, initial_state, HORIZON, STATES,
        T, beta)
    # convert state values to number of units done
    s_unit = np.where(s > 22, s-23, s)
    series.append(s_unit/2)
    timeseries_to_cluster.append(s_unit/(s_unit[-1]*2))

# cluster timeseries
km = TimeSeriesKMeans(n_clusters=3, n_init=5, metric="euclidean", verbose=True)
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
