"""
Functions for constructing the reward functions and transition matrices for
Zhang and Ma (2023) NYU study
also policy functions
"""
import numpy as np
from scipy.stats import binom


def reward_threshold(states, actions, reward_shirk, reward_thr,
                     reward_extra, thr, states_no):
    """
    reward function when units are rewarded immediately once threshold of
    thr no. of units are hit (compensated at reward_thr per unit) and then
    reward_extra per every extra unit until states_no (max no. of states) units
    in the course, thr=14 and max no of units = 22
    """

    reward_func = []
    for state_current in range(len(states)):

        reward_temp = np.zeros((len(actions[state_current]), len(states)))

        # rewards for shirk based on the action
        for action in range(len(actions[state_current])):

            reward_temp[action, 0:state_current+action+1] = (
                (len(states)-1-action) * reward_shirk)

        # if less than thr credits have been completed, then thresholded reward
        # at thr units, flatter extra rewards till states_no and then flat
        if state_current < thr:

            for i, action in enumerate(
                    actions[state_current][thr-state_current:
                                           states_no-state_current]):

                reward_temp[action, thr:action+state_current+1] += (
                    thr*reward_thr
                    + np.arange(0, action+state_current+1-thr, step=1)
                    * reward_extra)

            for i, action in enumerate(actions[state_current]
                                       [states_no-state_current:]):

                reward_temp[action, thr:states_no] += np.arange(
                    thr*reward_thr,
                    thr*reward_thr + (states_no-thr)*reward_extra,
                    step=reward_extra)
                reward_temp[action, states_no:action+state_current+1] += (
                    thr*reward_thr + (states_no-1-thr)*reward_extra)

        # if more than 14 units completed, extra reward unitl 22 is reached
        # and then nothing
        elif state_current >= thr and state_current < states_no-1:

            for i, action in enumerate(actions[state_current]
                                       [:states_no-state_current]):

                reward_temp[action, state_current+1:
                            action+state_current+1] += (
                                np.arange(1, action+1)*reward_extra)

            # reward_temp[states_no-state_current:, :] = reward_temp[
            # states_no-1-state_current, :]
        reward_func.append(reward_temp)

    return reward_func


def reward_immediate(states, actions, reward_shirk,
                     reward_unit, reward_extra):
    """
    reward function when units are rewarded immediately;
    shirk rewards are also immediate
    """

    reward_func = []

    for state_current in range(len(states)):

        reward_temp = np.zeros((len(actions[state_current]), len(states)))

        # rewards for shirk based on the action
        for action in range(len(actions[state_current])):

            reward_temp[action, 0:state_current+action+1] = (
                (len(states)-1-action) * reward_shirk)

        # immediate rewards for units completed
        for action in range(len(actions[state_current])):

            reward_temp[action, state_current:state_current+action+1] += (
                np.arange(0, action+1) * reward_unit
            )

        reward_func.append(reward_temp)

    return reward_func


def reward_no_immediate(states, actions, reward_shirk):
    """
    The only immediate rewards are from shirk
    """

    reward_func = []
    for state_current in range(len(states)):

        reward_temp = np.zeros((len(actions[state_current]), len(states)))

        # rewards for shirk based on the action
        for action in range(len(actions[state_current])):

            reward_temp[action, 0:state_current+action+1] = (
                (len(states)-1-action) * reward_shirk)

        reward_func.append(reward_temp)

    return reward_func


def effort(states, actions, effort_work):
    """
    immediate effort from actions
    """

    effort_func = []
    for state_current in range(len(states)):

        effort_temp = np.zeros((len(actions[state_current]), len(states)))

        for i, action in enumerate(actions[state_current]):

            effort_temp[action, :] = action * effort_work

        effort_func.append(effort_temp)

    return effort_func


def effort_convex_concave(states, actions, effort_work, exponent):
    """
    immediate effort from actions, allowing not only linear but also concave
    and convex costs as functions of number of units done
    """

    effort_func = []
    for state_current in range(len(states)):

        effort_temp = np.zeros((len(actions[state_current]), len(states)))

        for i, action in enumerate(actions[state_current]):

            effort_temp[action, :] = (action**exponent) * effort_work

        effort_func.append(effort_temp)

    return effort_func


def reward_final(states, reward_thr, reward_extra, thr, states_no):
    """
    when reward comes at final step -- again threshold at thr and extra rewards
    uptil states_NO (max number of states)
    in the course, thr=14 and max no of units = 22
    """
    total_reward_func_last = np.zeros(len(states))
    # np.zeros(len(states))
    # np.arange(0, states_no, 1)*reward_thr
    total_reward_func_last[thr:states_no] = (
        thr*reward_thr + np.arange(0, states_no-thr)*reward_extra)
    total_reward_func_last[states_no:] = (
        thr*reward_thr + (states_no-1-thr)*reward_extra)

    return total_reward_func_last


def T_binomial(states, actions, efficacy):
    """
    transition function as binomial number of successes with
    probability=efficacy for number of units worked  (=action)
    """

    T = []
    for state_current in range(len(states)):

        T_temp = np.zeros((len(actions[state_current]), len(states)))

        for i, action in enumerate(actions[state_current]):

            T_temp[action, state_current:state_current+action+1] = (
                binom(action, efficacy).pmf(np.arange(action+1))
            )

        T.append(T_temp)

    return T


def T_binomial_decreasing(states, actions, horizon, efficacy):
    """
    time-decreasing binomial transition probabilities
    """

    T = []
    for i_timestep in range(horizon):
        T_t = []
        efficacy_t = efficacy * (1 - (i_timestep / horizon))

        for state_current in range(len(states)):

            T_temp = np.zeros((len(actions[state_current]), len(states)))

            for i, action in enumerate(actions[state_current]):

                T_temp[action, state_current:state_current+action+1] = (
                    binom(action, efficacy_t).pmf(np.arange(action+1))
                )

            T_t.append(T_temp)
        T.append(T_t)
    return T

# define reward structure
# define separately for low and high rewards states and then extend


def generate_interest_rewards(states, states_no, actions_base,
                              reward_shirk, reward_unit, reward_extra,
                              effort_work, reward_interest):
    """
    reward structure for case where there are low and high reward super-states
    high reward state = base reward structure + interest rewards
    """

    # base course rewards (threshold of 14 units)
    reward_func_base = reward_threshold(
        states[:int(states_no/2)], actions_base, reward_shirk,
        reward_unit, reward_extra)

    # immediate interest rewards
    reward_func_interest = reward_immediate(
        states[:int(states_no/2)], actions_base, 0, reward_interest,
        reward_interest)

    # effort costs
    effort_func = effort(states[:int(states_no/2)], actions_base, effort_work)

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


def get_transitions_interest_states(states, states_no, actions_base, efficacy,
                                    p_stay_l, p_stay_h):
    """
    transition structure when there are low and high reward super-states
    define transitions separately within each super states and then combine
    """

    T_partial = T_binomial(states[:int(states_no/2)],
                           actions_base, efficacy)
    T_low = []
    for state_current in range(len(states[:int(states_no/2)])):

        temp = np.block([p_stay_l * T_partial[state_current],
                         (1 - p_stay_l) * T_partial[state_current]])
        # do probabilities sum to 1
        assert (np.round(np.sum(temp, axis=1), 6) == 1).all()
        T_low.append(temp)

    T_high = []
    for state_current in range(len(states[:int(states_no/2)])):

        temp = np.block([(1 - p_stay_h) * T_partial[state_current],
                         (p_stay_h) * T_partial[state_current]])
        # do probabilities sum to 1
        assert (np.round(np.sum(temp, axis=1), 6) == 1).all()
        T_high.append(temp)

    T = []
    T.extend(T_low)
    T.extend(T_high)

    return T


def deterministic_policy(a):
    """
    output determinsitic policy given Q-values
    """
    p = np.where(a == np.max(a), 1, 0)
    return p / sum(p)


def softmax_policy(a, beta):
    """
    output softmax actions given Q-values and inv temperature beta
    """
    c = a - np.max(a)
    p = np.exp(beta*c) / np.sum(np.exp(beta*c))
    return p
