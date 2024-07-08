"""
module for algorithms used to find optimal policy in Markov Decision
Processes (MDPs). The algorithms are based on dynamic programming. They cover
various scenarios: single or multiple discount factors, finite or infinite
horizons, does reward depend on the next state etc.
"""

import numpy as np


def find_optimal_policy(states, actions, horizon, discount_factor,
                        reward_func, reward_func_last, T):
    """
    find optimal policy for an MDP with finite horizon, discrete
    states, deterministic rewards and actions using dynamic programming

    params:
        states (ndarray): states of an MDP
        actions (list): actions available in each state
        horizon (int): number of timesteps in the task
        discount factor (float): factor by which rewards are diminished with
        delay
        reward_func (list): rewards on taking an action at a particular state
        reward_func_last (list): rewards on the final timestep only
        T (list): transition probabilities for each action in a state

    returns:
        V_opt (ndarray): optimal values of actions to take at each timestep
        and state
        policy_opt (ndarray): optimal action to take at each timestep and state
        Q_values (list): optimal values for each timestep, action and state

    """

    # arrays for optimal values, policy, Q-values
    V_opt = np.full((len(states), horizon+1), np.nan)
    policy_opt = np.full((len(states), horizon), np.nan)
    Q_values = np.full(len(states), np.nan, dtype=object)

    for i_state, state in enumerate(states):

        # V_opt for last time-step
        V_opt[i_state, -1] = reward_func_last[i_state]
        # arrays to store Q-values for each action in each state
        Q_values[i_state] = np.full((len(actions[i_state]), horizon), np.nan)

    # backward induction to derive optimal policy
    for i_timestep in range(horizon-1, -1, -1):

        for i_state, state in enumerate(states):

            Q = np.full(len(actions[i_state]), np.nan)

            for i_action, action in enumerate(actions[i_state]):

                # q-value for each action (bellman equation)
                Q[i_action] = (reward_func[i_state][i_action]
                               + discount_factor * (T[i_state][i_action]
                                                    @ V_opt[states,
                                                            i_timestep+1]))

            # find optimal action (which has max q-value)
            V_opt[i_state, i_timestep] = np.max(Q)
            policy_opt[i_state, i_timestep] = np.argmax(Q)
            Q_values[i_state][:, i_timestep] = Q

    return V_opt, policy_opt, Q_values


def find_optimal_policy_prob_rewards(states, actions, horizon, discount_factor,
                                     reward_func, reward_func_last, T):
    """
    find optimal policy for an MDP with finite horizon, discrete
    states, deterministic rewards and actions using dynamic programming.
    Now, reward recieved depends on the next state as well

    params:
        states (ndarray): states of an MDP
        actions (list): actions available in each state
        horizon (int): number of timesteps in the task
        discount factor (float): factor by which rewards are diminished with
        delay
        reward_func (list): rewards obtained on taking an action and
        transitioning to a particular state for each timestep and initial state
        reward_func_last (list): rewards on final timestep only
        T (list): transition probabilities for each action in a state

    returns:
        V_opt (ndarray): optimal values of actions to take at each timestep
        and state
        policy_opt (ndarray): optimal action to take at each timestep and state
        Q_values (list): optimal values for each timestep, action and state

    """

    # arrays for optimal values, policy, Q-values
    V_opt = np.full((len(states), horizon+1), np.nan)
    policy_opt = np.full((len(states), horizon), np.nan)
    Q_values = np.full(len(states), np.nan, dtype=object)

    for i_state, state in enumerate(states):

        # V_opt for last time-step
        V_opt[i_state, -1] = reward_func_last[i_state]
        # arrays to store Q-values for each action in each state
        Q_values[i_state] = np.full((len(actions[i_state]), horizon), np.nan)

    # backward induction to derive optimal policy
    for i_timestep in range(horizon-1, -1, -1):

        for i_state, state in enumerate(states):

            Q = np.full(len(actions[i_state]), np.nan)

            for i_action, action in enumerate(actions[i_state]):

                # q-value for each action (bellman equation)
                Q[i_action] = (T[i_state][i_action]
                               @ reward_func[i_state][i_action].T
                               + discount_factor * (T[i_state][i_action]
                                                    @ V_opt[states,
                                                            i_timestep+1]))

            # find optimal action (which gives max q-value)
            V_opt[i_state, i_timestep] = np.max(Q)
            policy_opt[i_state, i_timestep] = np.argmax(Q)
            Q_values[i_state][:, i_timestep] = Q

    return V_opt, policy_opt, Q_values


def find_optimal_policy_T_time_dep(states, actions, horizon, discount_factor,
                                   reward_func, reward_func_last, T):
    """
    params:
        states (ndarray): states of an MDP
        actions (list): actions available in each state
        horizon (int): number of timesteps in the task
        discount factor (float): factor by which rewards are diminished with
        delay
        reward_func (list): rewards obtained on taking an action and
        transitioning to a particular state for each timestep and initial state
        reward_func_last (list): rewards on final timestep only
        T (list): transition probabilities for each action in each timstep and
        state

    returns:
        V_opt (ndarray): optimal values of actions to take at each timestep
        and state
        policy_opt (ndarray): optimal action to take at each timestep and state
        Q_values (list): optimal values for each timestep, action and state


    """

    # arrays for optimal values, policy, Q-values
    V_opt = np.full((len(states), horizon+1), np.nan)
    policy_opt = np.full((len(states), horizon), np.nan)
    Q_values = np.full(len(states), np.nan, dtype=object)

    for i_state, state in enumerate(states):

        # V_opt for last time-step
        V_opt[i_state, -1] = reward_func_last[i_state]
        # arrays to store Q-values for each action in each state
        Q_values[i_state] = np.full((len(actions[i_state]), horizon), np.nan)

    # backward induction to derive optimal policy
    for i_timestep in range(horizon-1, -1, -1):

        for i_state, state in enumerate(states):

            Q = np.full(len(actions[i_state]), np.nan)

            for i_action, action in enumerate(actions[i_state]):

                # q-value for each action (bellman equation)
                Q[i_action] = (T[i_timestep][i_state][i_action]
                               @ reward_func[i_state][i_action].T
                               + discount_factor
                               * (T[i_timestep][i_state][i_action]
                                  @ V_opt[states, i_timestep+1]))

            # find optimal action (which gives max q-value)
            V_opt[i_state, i_timestep] = np.max(Q)
            policy_opt[i_state, i_timestep] = np.argmax(Q)
            Q_values[i_state][:, i_timestep] = Q

    return V_opt, policy_opt, Q_values


# my algorithm!!! for finding optimal policy with different discount factors
# for positive and negative rewards
def find_optimal_policy_diff_discount_factors(
        states, actions, horizon, discount_factor_reward, discount_factor_cost,
        reward_func, cost_func, reward_func_last, cost_func_last, T):
    """
    find optimal policy with different exponential discount
    factors for rewards and efforts, for a finite horizon and discrete states/
    actions; since the optimal policy can reverse in time, it is found starting
    at every timestep

     params:
         states (ndarray): states of an MDP
         actions (list): actions available in each state
         horizon (int): number of timesteps in the task
         discount_factor_reward (float): factor by which positive rewards are
         diminished with delay
         discount_factor_cost (float): factor by which negative rewards are
         diminished with delay
         reward_func (list): positive rewards obtained on taking an action and
         transitioning to a specific state for each timestep and initial state
         reward_func_last (list): positive rewards on final timestep only
         cost_func (list): negative rewards obtained on taking an action and
         transitioning to a specific state for each timestep and initial state
         cost_func_last (list): negative rewards on final timestep only
         T (list): transition probabilities for each action in a state

     returns:
         V_opt (ndarray): optimal values of actions to take at each timestep
         and state
         policy_opt (ndarray): map specifying optimal action to take at each
         timestep and state
         Q_values (list): optimal values for each timestep, action and state

    """

    V_opt_full = []
    policy_opt_full = []
    Q_values_full = []

    # solve for optimal policy at every time step
    for i_iter in range(horizon-1, -1, -1):

        V_opt = np.zeros((len(states), horizon+1))
        policy_opt = np.full((len(states), horizon), np.nan)
        Q_values = np.zeros(len(states), dtype=object)

        for i_state, state in enumerate(states):

            # V_opt for last time-step
            V_opt[i_state, -1] = ((discount_factor_reward**(horizon-i_iter))
                                  * reward_func_last[i_state]
                                  + (discount_factor_cost**(horizon-i_iter))
                                  * cost_func_last[i_state])
            # arrays to store Q-values for each action in each state
            Q_values[i_state] = np.full((len(actions[i_state]), horizon),
                                        np.nan)

        # backward induction to derive optimal policy starting from
        # timestep i_iter
        for i_timestep in range(horizon-1, i_iter-1, -1):

            for i_state, state in enumerate(states):

                Q = np.full(len(actions[i_state]), np.nan)

                for i_action, action in enumerate(actions[i_state]):

                    r = ((discount_factor_reward**(i_timestep-i_iter))
                         * reward_func[i_state][i_action]
                         + (discount_factor_cost**(i_timestep-i_iter))
                         * cost_func[i_state][i_action])

                    # q-value for each action (bellman equation)
                    Q[i_action] = (T[i_state][i_action] @ r.T
                                   + T[i_state][i_action]
                                   @ V_opt[states, i_timestep+1])

                # find optimal action (which gives max q-value)
                V_opt[i_state, i_timestep] = np.max(Q)
                policy_opt[i_state, i_timestep] = np.argmax(Q)
                Q_values[i_state][:, i_timestep] = Q

        V_opt_full.append(V_opt)
        policy_opt_full.append(policy_opt)
        Q_values_full.append(Q_values)

    return V_opt_full, policy_opt_full, Q_values_full


def forward_runs(policy, V, initial_state, horizon, states, T):
    """
    simulate actions taken and states reached forward in time given
    a policy and initial state in an mdp

    params:
        policy (ndarray): map of action to take at each timestep and state
        V (ndarray): value of corresponding policy in mdp
        initial_state (int): initial state that agent occupies
        horizon (int): number of timesteps in the task
        states (list): states of an MDP
        T (list): ransition probabilities for each action in a state

    returns:
        states_forward (ndarray): sequence of states visited in time
        actions_forward (ndarray): sequence of actions taken in time
        values_forward (ndarray): corresponding state values

    """

    # arrays to store states, actions taken and values of actions in time
    states_forward = np.full(horizon+1, 100)
    actions_forward = np.full(horizon, 100)
    values_forward = np.full(horizon, np.nan)

    states_forward[0] = initial_state

    for i_timestep in range(horizon):

        # action at a state and timestep as given by policy
        actions_forward[i_timestep] = policy[states_forward[i_timestep],
                                             i_timestep]
        # corresponding value
        values_forward[i_timestep] = V[states_forward[i_timestep], i_timestep]
        # next state given by transition probabilities
        states_forward[i_timestep+1] = np.random.choice(
            len(states),
            p=T[states_forward[i_timestep]][actions_forward[i_timestep]]
        )

    return states_forward, actions_forward, values_forward


def forward_runs_prob(policy, Q_values, actions, initial_state, horizon,
                      states, T, *args):
    """
    simulate actions taken and states reached forward in time given Q-values
    corresponding to a policy and initial state in an mdp; here, there is a
    probabilistic policy function that takes Q-values and returns actions

    params:
        policy (function): probabilistic function that returns actions given
        Q-values of actions in a policy
        Q_values (ndarray): Q-values of actions that can be taken at each
        timestep and in each state, according to a policy map
        actions (list):  actions available in each state
        initial_state (int): initial state that agent occupies
        horizon (int): number of timesteps in the task
        states (list): states of an MDP
        T (list): ransition probabilities for each action in a state

    returns:
        states_forward (ndarray): sequence of states visited in time
        actions_forward (ndarray): sequence of actions taken in time
    """

    # arrays to store states, actions taken and values of actions in time
    states_forward = np.full(horizon+1, 100)
    actions_forward = np.full(horizon, 100)

    states_forward[0] = initial_state

    for i_timestep in range(horizon):

        actions_forward[i_timestep] = np.random.choice(
            actions[states_forward[i_timestep]],
            p=policy(
                Q_values[states_forward[i_timestep]][:, i_timestep], args)
        )

        # next state given by transition probabilities
        states_forward[i_timestep+1] = np.random.choice(
            len(states),
            p=T[states_forward[i_timestep]][actions_forward[i_timestep]]
        )

    return states_forward, actions_forward


def forward_runs_T_time_dep(policy, Q_values, actions, initial_state, horizon,
                            states, T, *args):
    """
    simulate actions taken and states reached forward in time given
    Q_values corresponding to a policy, nitial state and probabilistic policy
    function; here transition probabilities are also time-dependant

    params:
        policy (function): probabilistic function that returns actions given
        Q-values of actions in a policy
        Q_values (ndarray): Q-values of actions that can be taken at each
        timestep and in each state, according to a policy map
        actions (list):  actions available in each state
        initial_state (int): initial state that agent occupies
        horizon (int): number of timesteps in the task
        states (list): states of an MDP
        T (list): ransition probabilities for each action in a state

    returns:
        states_forward (ndarray): sequence of states visited in time
        actions_forward (ndarray): sequence of actions taken in time
    """

    # arrays to store states, actions taken and values of actions in time
    states_forward = np.full(horizon+1, 100)
    actions_forward = np.full(horizon, 100)

    states_forward[0] = initial_state

    for i_timestep in range(horizon):

        if args:
            # action at a state and timestep as given by policy
            actions_forward[i_timestep] = np.random.choice(
                actions[states_forward[i_timestep]],
                p=policy(
                    Q_values[states_forward[i_timestep]][:, i_timestep], args)
            )
        else:
            # action at a state and timestep as given by policy
            actions_forward[i_timestep] = np.random.choice(
                actions[states_forward[i_timestep]],
                p=policy(
                    Q_values[states_forward[i_timestep]][:, i_timestep])
            )

        # next state given by transition probabilities
        states_forward[i_timestep+1] = np.random.choice(
            len(states),
            p=T[i_timestep][
                states_forward[i_timestep]][actions_forward[i_timestep]]
        )

    return states_forward, actions_forward
