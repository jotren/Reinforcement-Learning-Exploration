import matplotlib.pyplot as plt
import numpy as np

# Parameters for the gambler's problem
goal = 100  # The target amount in pounds
p_heads = 0.4  # Probability of heads
states = np.arange(goal + 1)  # States are the gambler's current capital: 0, 1, ..., 100
policy = np.zeros(goal + 1)  # Initial policy (arbitrary)

print(states)

value_function = np.zeros(goal + 1)  # Value function, initialized to 0 for all states
value_function[goal] = 1  # The value of the goal state is set to 1

print(value_function)

# This function we are using the Bellman Equation to calcualte the maximum return from all possible bets (0->100)
# For each possible state. This is the Policy Evaluation Step
def policy_evaluation(policy, V, theta=1e-8, discount_factor=1.0):
    while True:
        delta = 0
        #loop through each value in the value function
        #s here represents a step from 0 - 100
        for s in range(1, goal):
            old_value = V[s]
            action_returns = []
            # This for loop ensures that the bet would not go above £100
            for a in range(1, min(s, goal - s) + 1):  # The bet can be from 1 to min(s, goal - s)
                # Calculate the expected value for a win and a loss
                win = p_heads * V[s + a]
                loss = (1 - p_heads) * V[s - a]
                action_returns.append(win + loss)
            # Update the value for the state
            V[s] = max(action_returns)
            delta = max(delta, abs(old_value - V[s]))
        # Check if the change in value function is below the threshold
        if delta < theta:
            break
    return V

# This function then uses the value function 
def policy_improvement(V, policy, discount_factor=1.0):
    policy_stable = True
    for s in range(1, goal):
        old_action = policy[s]
        action_returns = []
        for a in range(1, min(s, goal - s) + 1):  # The bet can be from 1 to min(s, goal - s)
            # Calculate the expected value for a win and a loss
            win = p_heads * V[s + a]
            loss = (1 - p_heads) * V[s - a]
            action_returns.append(win + loss)
        # Find the action with the highest expected return
        print(action_returns)
        best_action = np.argmax(action_returns) + 1  # Adding 1 because actions are 1-indexed
        policy[s] = best_action
        if old_action != best_action:
            policy_stable = False
    return policy, policy_stable

# Policy Iteration
for x in range(0,1):
    # Evaluate the current policy
    value_function = policy_evaluation(policy, value_function)
    # Improve the policy
    policy, policy_stable = policy_improvement(value_function, policy)
    # If the policy is stable we've found an optimal policy
    if policy_stable:
        break

# Plot the final optimal policy, which indicates the stake (bet) for each state
plt.figure(figsize=(10, 6))
plt.bar(np.arange(goal + 1), policy, align='center', alpha=0.7)
plt.title('Optimal Policy (Stake Size at Each Capital Level)')
plt.xlabel('Capital')
plt.ylabel('Stake (Bet Size)')
plt.xlim([1, goal - 1])  # Omit the terminal states (0 and 100) from the plot
plt.grid()
plt.show()



# # Plot the final converged value function
# plt.plot(value_function, label='Optimal Value Function')
# plt.title('Final Converged Value Function')
# plt.xlabel('Capital')
# plt.ylabel('Value Estimates')
# plt.legend()
# plt.grid()
# plt.show()

