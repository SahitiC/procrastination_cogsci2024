"""
module for constructing the reward functions and transition matrices for
the task in Zhang and Ma (2023) NYU study
"""
import numpy as np
from scipy.special import comb


def reward_threshold(states, actions, reward_shirk, reward_thr,
                     reward_extra, thr, states_no):
    """
    construct reward function where units are rewarded immediately once
    threshold no. of units are hit (compensated at reward_thr per unit) & then
    reward_extra per every extra unit until max no. of states units
    (in Zhang and Ma data, thr=14 and max no of units = 22); reward for
    shirking is immediate

    params:
        states (ndarray): states of an MDP
        actions (list): actions available in each state
        reward_shirk (float): reward for doing an alternate task (i.e., for
        each unit that is not used for work)
        reward_thr (float): reward for each unit of work completed until thr
        reward_extra (float): reward for each unit completed beyond thr
        thr (int): threshold number of units until which no reward is obtained
        states_no (int): max. no of units that can be completed

    returns:
        reward_func (list[ndarray]): rewards at each time point on taking each
        action at each state

    """

    reward_func = []
    for state_current in range(len(states)):

        reward_temp = np.zeros((len(actions[state_current]), len(states)))

        # rewards for shirk based on the action
        for action in range(len(actions[state_current])):

            reward_temp[action, state_current:state_current+action+1] = (
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

        # if more than 14 units completed, extra reward until 22 is reached
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


def reward_immediate(states, actions, reward_shirk, reward_unit):
    """
    construct reward function where units are rewarded immediately (compensated
    at reward_unit per unit); reward for shirking is also immediate

    params:
        states (ndarray): states of an MDP
        actions (list): actions available in each state
        reward_shirk (float): reward for doing an alternate task (i.e., for
        each unit that is not used for work)
        reward_unit (float): reward for each unit of work completed

    returns:
        reward_func (list[ndarray]): rewards at each time point on taking each
        action at each state

    """

    reward_func = []

    for state_current in range(len(states)):

        reward_temp = np.zeros((len(actions[state_current]), len(states)))

        # rewards for shirk based on the action
        for action in range(len(actions[state_current])):

            reward_temp[action, state_current:state_current+action+1] = (
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
    construct reward function where only reward for shirking is immediate

    params:
        states (ndarray): states of an MDP
        actions (list): actions available in each state
        reward_shirk (float): reward for doing an alternate task (i.e., for
        each unit that is not used for work)

    returns:
        reward_func (list[ndarray]): rewards at each time point on taking each
        action at each state

    """

    reward_func = []
    for state_current in range(len(states)):

        reward_temp = np.zeros((len(actions[state_current]), len(states)))

        # rewards for shirk based on the action
        for action in range(len(actions[state_current])):

            reward_temp[action, state_current:state_current+action+1] = (
                (len(states)-1-action) * reward_shirk)

        reward_func.append(reward_temp)

    return reward_func


def effort(states, actions, effort_work):
    """
    construct effort function (effort is always immediate)

    params:
        states (ndarray): states of an MDP
        actions (list): actions available in each state
        effort_work (float): cost for working per unit work

    returns:
        effort_func (list[ndarray]): effort at each time point on taking each
        action at each state
    """

    effort_func = []
    for state_current in range(len(states)):

        effort_temp = np.zeros((len(actions[state_current]), len(states)))

        for i, action in enumerate(actions[state_current]):

            effort_temp[action, state_current:state_current +
                        action+1] = action * effort_work

        effort_func.append(effort_temp)

    return effort_func


def effort_convex_concave(states, actions, effort_work, exponent):
    """
    construct effort function where cost per unit changes as an exponent in
    no. of units (e = effort_work * actions^exponent)

    params:
        states (ndarray): states of an MDP
        actions (list): actions available in each state
        effort_work (float): cost for working per unit work
        exponent (float): >1, defines convexity of effort function

    returns:
        effort_func (list[ndarray]): effort at each time point on taking each
        action at each state
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
    construct reward function where units are rewarded at end of task
    (compensated at reward_thr per unit)

    params:
        states (ndarray): states of an MDP
        reward_thr (float): reward for each unit of work completed until thr
        reward_extra (float): reward for each unit completed beyond thr
        thr (int): threshold number of units until which no reward is obtained
        states_no (int): max. no of units that can be completed

    returns:
        reward_func (list[ndarray]): rewards at each time point on taking each
        action at each state

    """
    total_reward_func_last = np.zeros(len(states))
    # np.zeros(len(states))
    # np.arange(0, states_no, 1)*reward_thr
    total_reward_func_last[thr:states_no] = (
        thr*reward_thr + np.arange(0, states_no-thr)*reward_extra)
    total_reward_func_last[states_no:] = (
        thr*reward_thr + (states_no-1-thr)*reward_extra)

    return total_reward_func_last


def binomial_pmf(n, p, k):
    """
    calculates binomial probability mass function

    params:
        n (int): number of trials
        p (float): (0<=p<=1) probability of success
        k (int): number of successes

    returns:
        binomial_prob (float): binomial probability given parameters
    """

    if not isinstance(n, (int, np.int32, np.int64)):
        print(type(n))
        raise TypeError("Input must be an integer.")

    binomial_prob = comb(n, k) * p**k * (1-p)**(n-k)

    return binomial_prob


def T_binomial(states, actions, efficacy):
    """
    transition function as binomial number of successes with
    probability=efficacy for number of units worked  (=action)

    params:
        states (ndarray): states of an MDP
        actions (list): actions available in each states
        efficacy (float): (0<=efficacy<=1) binomial probability of success on
                          doing some units of work (action)

    returns:
        T (list[ndarray]): transition matrix
    """

    T = []
    for state_current in range(len(states)):

        T_temp = np.zeros((len(actions[state_current]), len(states)))

        for i, action in enumerate(actions[state_current]):

            T_temp[action, state_current:state_current+action+1] = (
                binomial_pmf(action, efficacy, np.arange(action+1))
            )

        T.append(T_temp)

    return T


def generate_interest_rewards(states, states_no, actions_base,
                              reward_shirk, reward_unit, reward_extra,
                              effort_work, reward_interest, thr):
    """
    reward function for case where there are low and high reward super-states
    high reward state = base reward structure + interest rewards

    params:
        states (ndarray): states of an MDP
        states_no (int): number of states
        actions_base (list): actions available for states within a super-state
        reward_shirk (float): reward for shirking
        reward_unit (float): reward for each unit of work completed until thr
        reward_extra (float): reward for each unit completed beyond thr
        effort_work (float): cost for working per unit work
        reward_interest (float): reward obtained per unit work on top of
                                reward_unit in high reward state
        thr (int): threshold number of units until which no reward is obtained

    returns:
        total_reward_func (list[ndarray]): total rewards at each time point on
        taking each action at each state
        total_reward_func_last (list[ndarray]): total rewards at the very last
        timestep for each state
    """

    # base course rewards (threshold of 14 units)
    reward_func_base = reward_threshold(
        states[:int(states_no/2)], actions_base, reward_shirk,
        reward_unit, reward_extra, thr, states_no)

    # immediate interest rewards
    reward_func_interest = reward_immediate(
        states[:int(states_no/2)], actions_base, reward_shirk,
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
    transition function when there are low and high reward super-states
    define transitions separately within each super states and then combine

    params:
        states (ndarray): states of an MDP
        states_no (int): number of states
        actions_base (list): actions available for states within a super-state
        efficacy (float): (0<=efficacy<=1) binomial probability of success on
                          doing some units of work (action)
       p_stay_l (float): probability of staying in low reward state
       p_stay_h (float): probability of staying in high reward state

    returns:
        T (list[ndarray]): transition matrix
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

    params:
        a (ndarray): array of q-values

    returns:
        (ndarray): choose action with best q-value and assign probability = 1
    """
    p = np.where(a == np.max(a), 1, 0)
    return p / sum(p)


def softmax_policy(a, beta):
    """
    output softmax actions given Q-values (a) and inv temperature beta

    params:
        a (ndarray): array of q-values
        beta (float): inverse temperature

    returns:
        p (ndarray): assign probabilities to action by applying softmax to
        q-values
    """
    c = a - np.max(a)
    p = np.exp(beta*c) / np.sum(np.exp(beta*c))
    return p
