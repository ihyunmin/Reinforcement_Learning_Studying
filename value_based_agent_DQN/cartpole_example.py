import gym
import math
env = gym.make('CartPole-v1')

# new criteria -90 degree ~ 90 degree.
def another_criteria(observation):
    done = False
    if observation[2] < math.radians(-90) or observation[2] > math.radians(90):
        done = True
    reward = 1.0
    return reward, done

for i_episode in range(50):
    observation = env.reset()
    for t in range(100):
        env.render()
        print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        print(reward, done)
        # reward, done = another_criteria(observation)
        
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break

env.close()