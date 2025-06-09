import gym 
import numpy as np


env = gym.make("FrozenLake-v1", is_slippery=False)
n_states = env.observation_space.n
n_actions = env.action_space.n
q_table = np.zeros((n_states, n_actions))
alpha = 0.8
gamma = 0.95
epsilon = 1.0
epsilon_min = 0.1
epsilon_decay = 0.995
episodes = 5000

for episode in range(1, episodes+1):
    state,_ = env.reset()
    total_reward = 0
    done = False
    while not done:
        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        old_value = q_table[state,action]
        next_max = np.max(q_table[next_state])
        q_table[state,action] = (1- alpha)* old_value + alpha * (reward + gamma*next_max)
        state = next_state
        total_reward += reward
    epsilon = max(epsilon_min, epsilon*epsilon_decay)
    if episode%500 == 0:
        print(f"Episode: {episode}, Reward: {total_reward}")
        print(q_table)    
