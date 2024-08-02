import numpy as np
import plotly.express as px
import plotly.io as pio
pio.renderers.default='browser'
import pandas as pd

goal = 100
policy = np.zeros(goal+1)
states= np.arange(goal+1)

value_function = np.zeros(goal+1)
value_function[goal] = 1

discount_factor = 1.0

p_heads = 0.5
p_tails = 1-p_heads

#This is optimal policy evaluation

# Going through every state and calculating the maximum value function
def policy_evalutaion(V, theta, states, discount_factor):
    
    while True:
        delta=0
        for s in states[1:goal]:
            V_old = V[s]
            action_returns = []
            for a in range(1,min(s, goal-s) + 1):
                win = p_heads * V[s+a]
                loss = p_tails * V[s-a]
                action_returns.append(win+loss)
                
            V[s] = max(action_returns)
            delta = max(delta, abs(V_old - V[s]))
        if delta < theta:
            break
    
    return V
                
#This is optimal policy improvement

# Going through every state and finding the action that gives you the maximum value function
def policy_improvement(V, policy, states, discount_factor):
    
    policy_stable = True
    for s in states[1:goal]:
        old_action = policy[s]
        action_returns = []
        for a in range(1,min(s, goal-s) + 1):
            win = p_heads * V[s+a]
            loss = p_tails * V[s-a]
            action_returns.append(win+loss)
            
        policy[s] = np.argmax(action_returns) + 1
        best_action = policy[s]
        if old_action != best_action:
            policy_stable = False
    
    return policy, policy_stable


policy_stable = False

while not policy_stable:

    value_function = policy_evalutaion(value_function, 1e-8, states, discount_factor)
    policy, policy_stable = policy_improvement(value_function,policy, states, discount_factor)


# Convert arrays to Pandas DataFrame for plotting
value_function_df = pd.DataFrame({
    'Capital': states,
    'Value': value_function
})

policy_df = pd.DataFrame({
    'Capital': states,
    'Policy': policy
})

# Plot the value function
value_function_fig = px.line(value_function_df, x='Capital', y='Value', title='Value Function')
value_function_fig.show()

# Plot the policy
policy_fig = px.bar(policy_df, x='Capital', y='Policy', title='Policy')
policy_fig.show()
