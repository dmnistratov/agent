import numpy as np
import gym

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from rl.agents import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory

# Define the environment
env = gym.make('PaperIo-v0')
np.random.seed(0)
env.seed(0)

# Define the neural network model
model = Sequential()
model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
model.add(Dense(16, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(env.action_space.n, activation='linear'))
print(model.summary())

# Define the memory
memory = SequentialMemory(limit=10000, window_length=1)

# Define the policy
policy = EpsGreedyQPolicy(eps=0.1)

# Define the agent
dqn = DQNAgent(model=model, memory=memory, policy=policy, 
               nb_actions=env.action_space.n, nb_steps_warmup=1000,  
               target_model_update=1e-2)

dqn.compile(Adam(lr=1e-3), metrics=['mae'])

# Train the agent for 5000 timesteps
dqn.fit(env, nb_steps=5000, visualize=False, verbose=2)

# Test the agent
dqn.test(env, nb_episodes=5, visualize=False)
