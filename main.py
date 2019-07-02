import gym
import time
env = gym.make('CartPole-v1')
env.reset()

while True:
    env.step(env.action_space.sample())
    env.render(mode='rgb_array')
    time.sleep(.01)
env.close()