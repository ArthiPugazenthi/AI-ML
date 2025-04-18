# -*- coding: utf-8 -*-
"""Reinforcement_Learning.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/18NDWa07cNokr2nq1okKvyO4g5T_JArB7

Import Libraries
"""

import numpy as np
import random
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler
import pickle

"""Load the modified dataset"""

file_path = "trading-1.csv"
df = pd.read_csv(file_path)

"""Normalize feature columns"""

scaler = StandardScaler()
feature_columns = [col for col in df.columns if col not in ['adjcp', 'daily_return', 'unnamed: 0', 'datadate', 'tic']]
df[feature_columns] = scaler.fit_transform(df[feature_columns])

"""Selecting features using Mutual Information"""

mi_scores = mutual_info_regression(df[feature_columns], df['adjcp'])
selected_features = [feature_columns[i] for i in np.argsort(mi_scores)[-6:]]  # Select top 6 features

"""Plot Mutual Information Scores as Bar Plot"""

plt.figure(figsize=(10, 6))
sns.barplot(x=mi_scores, y=feature_columns, palette="viridis")
plt.xlabel("Mutual Information Score")
plt.ylabel("Feature")
plt.title("Feature Importance based on Mutual Information")
plt.show()

print("Selected Features for RL:", selected_features)

actions = ['buy', 'sell', 'hold']  # Possible actions
state_size = len(selected_features)  # Dynamic state size based on feature selection
action_size = len(actions)

"""Define a function to extract state representation"""

def get_state(index):
    return np.array(df.loc[index, selected_features])

"""Initialize Q-table"""

q_table = np.zeros((len(df), action_size))

"""Hyperparameters"""

learning_rate = 0.1
discount_factor = 0.95
epsilon = 1.0  # Exploration-exploitation trade-off
epsilon_decay = 0.995
min_epsilon = 0.01
transaction_cost = 0.001  # Simulated transaction cost per trade
checkpoint_interval = 10  # Save Q-table every 10 episodes

"""Function to choose an action"""

def choose_action(state_index, epsilon):
    if random.uniform(0, 1) < epsilon:
        return random.choice(actions)  # Explore
    else:
        return actions[np.argmax(q_table[state_index])]  # Exploit

"""Function to update Q-table"""

def update_q_table(state_index, action, reward, next_state_index):
    action_index = actions.index(action)
    best_next_action = np.argmax(q_table[next_state_index])
    q_table[state_index, action_index] = (1 - learning_rate) * q_table[state_index, action_index] + \
                                         learning_rate * (reward + discount_factor * q_table[next_state_index, best_next_action])

"""Function to calculate reward"""

def calculate_reward(state_index, action, next_state_index):
    price_diff = df.loc[next_state_index, 'adjcp'] - df.loc[state_index, 'adjcp']
    volatility_penalty = abs(df.loc[next_state_index, 'daily_return']) * 0.1  # Penalizing high volatility

    if action == 'buy':
        reward = price_diff - transaction_cost  # Profit minus transaction cost
    elif action == 'sell':
        reward = -price_diff - transaction_cost  # Negative reward for price increase after selling
    else:
        reward = -volatility_penalty  # Small penalty for holding during high volatility

    return reward

"""Training loop"""

num_episodes = 100
total_rewards = []
best_reward = float('-inf')
best_q_table = None
best_episode = -1

for episode in range(num_episodes):
    state_index = random.randint(0, len(df) - 2)  # Start at a random position
    done = False
    total_reward = 0

    while not done:
        action = choose_action(state_index, epsilon)
        next_state_index = state_index + 1 if state_index + 1 < len(df) else state_index  # Move to next step
        reward = calculate_reward(state_index, action, next_state_index)
        update_q_table(state_index, action, reward, next_state_index)
        total_reward += reward
        state_index = next_state_index

        if state_index >= len(df) - 1:
            done = True

    epsilon = max(min_epsilon, epsilon * epsilon_decay)  # Decay epsilon
    total_rewards.append(total_reward)

    if total_reward > best_reward:
        best_reward = total_reward
        best_q_table = np.copy(q_table)
        best_episode = episode + 1  # Store best episode index

    if episode % checkpoint_interval == 0:
        with open(f"q_table_checkpoint_{episode}.pkl", "wb") as f:
            pickle.dump(q_table, f)
        print(f"Checkpoint saved at episode {episode}")

    if episode % 10 == 0:
        print(f"Episode {episode + 1}, Total Reward: {total_reward}")

# Save the best Q-table
with open("best_q_table.pkl", "wb") as f:
    pickle.dump(best_q_table, f)

print(f"Best Q-table saved with highest reward: {best_reward}, from Episode: {best_episode}")

"""Plot Reward Trend"""

plt.figure(figsize=(10, 6))
plt.plot(range(1, num_episodes + 1), total_rewards, marker='o', linestyle='-', color='b')
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Reward Trend Over Episodes")
plt.grid()
plt.show()