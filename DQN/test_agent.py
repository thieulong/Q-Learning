import gym
import keras
import numpy as np

# Initialize environment
env = gym.make("CartPole-v1", render_mode = "human") # Show how agent play the game
state, _ = env.reset()
state_size = env.observation_space.shape[0]

# Load trained agent
my_agent = keras.models.load_model("trained_agent.h5")
n_timesteps = 200
total_reward = 0

for t in range(n_timesteps):
    env.render()
    # Take current state to predict
    state = state.reshape((1, state_size))
    q_values = my_agent.predict(state, verbose=0)
    max_q_values = np.argmax(q_values)

    next_state, reward, terminal, _, _ = env.step(action=max_q_values)
    total_reward += reward
    state = next_state
    print(f"Timestep: {t}")