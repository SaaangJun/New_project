import gym
import pressureplate

env = gym.make('pressureplate-linear-3p-v0')
env.reset()

for i in range(1000):
    env.render()
    env.step(env.action_space.sample())
