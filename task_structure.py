"""
Functions for constructing the reward functions and transition matrices for
Zhang and Ma (2023) NYU study
"""
import numpy as np
from scipy.stats import binom


def reward_threshold(states, actions, reward_shirk, reward_thr,
                     reward_extra):
    """
    reward function when units are rewarded immediately once threshold of
    14 units are hit (compensated at reward_thr per unit) and then reward_extra
    per every extra unit until 22 units
    """

    reward_func = []
    for state_current in range(len(states)):

        reward_temp = np.zeros((len(actions[state_current]), len(states)))

        # rewards for shirk based on the action
        for action in range(len(actions[state_current])):

            reward_temp[action, 0:state_current+action+1] = (
                (len(states)-1-action) * reward_shirk)

        # if less than 14 credits have been completed, then thresholded reward
        # at 14 units, flatter extra rewards till 22 and then flat
        if state_current < 14:

            for i, action in enumerate(actions[state_current]
                                       [14-state_current:22-state_current+1]):

                reward_temp[action, 14:action+state_current+1] += (
                    14*reward_thr
                    + np.arange(0, action+state_current+1-14, step=1)
                    * reward_extra)

            for i, action in enumerate(actions[state_current]
                                       [22-state_current+1:]):

                reward_temp[action, 14:23] += np.arange(
                    14*reward_thr, 14*reward_thr + (22-14+1)*reward_extra,
                    step=reward_extra)
                reward_temp[action, 23:action+state_current+1] += (
                    14*reward_thr + (22-14)*reward_extra)

        # if more than 14 units completed, extra reward unitl 22 is reached
        # and then nothing
        elif state_current >= 14 and state_current < 22:

            for i, action in enumerate(actions[state_current]
                                       [:22-state_current+1]):

                reward_temp[action, state_current+1:
                            action+state_current+1] += (
                                np.arange(1, action+1)*reward_extra)

            # reward_temp[22-state_current+1:, :] = reward_temp[22-state_current, :]
        reward_func.append(reward_temp)

    return reward_func


def reward_immediate(states, actions, reward_shirk,
                     reward_unit, reward_extra):

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


def reward_final(states, reward_thr, reward_extra):
    """
    when reward comes at final step -- again threshold at 14 and extra rewards
    uptil 22
    """
    total_reward_func_last = np.zeros(len(states))
    # np.zeros(len(states))
    # np.arange(0, states_no, 1)*reward_thr
    total_reward_func_last[14:22+1] = (14*reward_thr
                                       + np.arange(0, 22-14+1)*reward_extra)
    total_reward_func_last[23:] = 14*reward_thr + (22-14)*reward_extra

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
