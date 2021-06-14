from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.environment import ActionTuple
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel

import numpy as np
import mlagents.trainers

from collections import namedtuple

obs = namedtuple(
        'obs',
        ['vector', 'front', 'right', 'back', 'left', 'raycast'])

def rgb2gray(rgb):

    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

class Drone(object):

    def __init__(self, time_scale=1.0, filename='mac.app', port=11000):

        self.engine_configuration_channel = EngineConfigurationChannel()
        print(f"VERSION : {mlagents.trainers.__version__}")
        self.env = UnityEnvironment(
                file_name=filename,
                worker_id=port,
                side_channels=[self.engine_configuration_channel])
        self.env.reset()

        self.behavior_name = list(self.env.behavior_specs.keys())[0]
        self.spec = self.env.behavior_specs[self.behavior_name]
        self.engine_configuration_channel.set_configuration_parameters(time_scale=time_scale)
        self.dec, self.term = self.env.get_steps(self.behavior_name)

    def reset(self):

        self.env.reset()
        self.dec, self.term = self.env.get_steps(self.behavior_name)
        self.tracked_agent = -1

        self.front = np.zeros([36, 64, 3])
        self.right = np.zeros([36, 64, 3])
        self.back = np.zeros([36, 64, 3])
        self.left = np.zeros([36, 64, 3])

        self.state = [self.dec.obs[i][0] for i in range(6)]

        self.front[:, :, -1] = rgb2gray(self.state[1])
        self.right[:, :, -1] = rgb2gray(self.state[2])
        self.back[:, :, -1] = rgb2gray(self.state[3])
        self.left[:, :, -1] = rgb2gray(self.state[4])

        self.state = obs(
                vector=self.state[0], front=self.front,
                right=self.right, back=self.back,
                left=self.left, raycast=self.state[5])

        return self.state

    def step(self, action):

        if self.tracked_agent == -1 and len(self.dec) >= 1:
            self.tracked_agent = self.dec.agent_id[0]

        action = np.clip(action, -1, 1)
        action_tuple = ActionTuple()
        action_tuple.add_continuous(np.array([action]))

        self.env.set_actions(self.behavior_name, action_tuple)
        self.env.step()

        self.dec, self.term = self.env.get_steps(self.behavior_name)

        reward = 0
        done = False
        if self.tracked_agent in self.dec:
            reward += self.dec[self.tracked_agent].reward
        if self.tracked_agent in self.term:
            reward += self.term[self.tracked_agent].reward
            done = True

        if done:
            return self.state, reward, done

        self.state = [self.dec.obs[i][0] for i in range(6)]

        self.front[:, :, :-1] = self.front[:, :, 1:]
        self.right[:, :, :-1] = self.right[:, :, 1:]
        self.back[:, :, :-1] = self.back[:, :, 1:]
        self.left[:, :, :-1]  = self.left[:, :, 1:]

        self.front[:, :, -1] = rgb2gray(self.state[1])
        self.right[:, :, -1] = rgb2gray(self.state[2])
        self.back[:, :, -1] = rgb2gray(self.state[3])
        self.left[:, :, -1] = rgb2gray(self.state[4])

        self.state = obs(
                vector=self.state[0], front=self.front,
                right=self.right, back=self.back,
                left=self.left, raycast=self.state[5])

        return self.state, reward, done

if __name__ == '__main__':

    import matplotlib.pyplot as plt

    env = Drone(
            time_scale=0.1,
            filename='/Users/chageumgang/Desktop/baselines/mac.app')
    
    episode = 0
    while True:

        state = env.reset()
        done = False
        score = 0
        episode += 1

        while not done:

            action = np.random.rand(3)
            next_state, reward, done = env.step(action)
            score += reward

            print(next_state.vector.shape)
            print(next_state.raycast.shape)
            print(next_state.front.shape)

            '''
            fig = plt.figure()
            ax1 = fig.add_subplot(2, 2, 1)
            ax1.imshow(state.front)
            ax1 = fig.add_subplot(2, 2, 2)
            ax1.imshow(state.back)
            ax1 = fig.add_subplot(2, 2, 3)
            ax1.imshow(state.right)
            ax1 = fig.add_subplot(2, 2, 4)
            ax1.imshow(state.left)
            plt.show(block=False)
            plt.pause(0.1)
            '''

            state = next_state

        print(episode, score)
