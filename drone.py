from baselines.env.drone import Drone
from baselines.agent.drone_impala import DroneAgent
from baselines.distributed_queue.drone_impala import Traj

from tensorboardX import SummaryWriter

import tensorflow as tf
import numpy as np

def main():

    trajectory = 4

    env = Drone(
            time_scale=1.0,
            filename='/Users/chageumgang/Desktop/baselines/mac.app')

    writer = SummaryWriter()

    episode = 0
    score = 0
    episode_step = 0
    prob = 0
    
    state = env.reset()

    actor = DroneAgent(
            trajectory=trajectory)
    sess = tf.Session()
    actor.set_session(sess)

    for _ in range(3):

        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(1,3,1)
        ax.imshow(state.front[:, :, 0])
        ax = fig.add_subplot(1,3,2)
        ax.imshow(state.front[:, :, 1])
        ax = fig.add_subplot(1,3,3)
        ax.imshow(state.front[:, :, 2])
        plt.show()

        state, reward, done = env.step([1, 0, 0])

    '''
    train_step = 0

    while True:

        unrolled_data = Traj(traj=trajectory)

        for _ in range(trajectory):

            action_tuple = actor.get_policy_and_action(
                    vector=state.vector, front=state.front,
                    right=state.right, back=state.back,
                    left=state.left, raycast=state.raycast)

            real_action = [
                    action_tuple.action_x - 1,
                    action_tuple.action_y - 1,
                    action_tuple.action_z - 1]

            next_state, reward, done = env.step(real_action)

            if reward == 100:
                reward = 1
            elif reward == -100:
                reward = -1
            elif reward == 0:
                reward = 0
            elif reward > 0:
                reward = 0.1
            elif reward < 0:
                reward = -0.1

            if episode_step > 300:
                reward = -1
                done = True

            episode_step += 1
            score += reward
            prob += action_tuple.prob_x[action_tuple.action_x] * \
                    action_tuple.prob_y[action_tuple.action_y] * \
                    action_tuple.prob_z[action_tuple.action_z]

            unrolled_data.append(
                    vector=state.vector, front=state.front,
                    right=state.right, back=state.back,
                    left=state.left, raycast=state.raycast,
                    reward=reward, done=done,
                    action_x=action_tuple.action_x, action_y=action_tuple.action_y,
                    action_z=action_tuple.action_z, mu_x=action_tuple.prob_x,
                    mu_y=action_tuple.prob_y, mu_z=action_tuple.prob_z)

            state = next_state

            print(reward, score, done)

            if done:

                print(episode, score)

                writer.add_scalar('data/score', score, episode)
                writer.add_scalar('data/prob', prob / episode_step, episode)
                writer.add_scalar('data/episode_step', episode_step, episode)

                state = env.reset()
                episode += 1
                score = 0
                episode_step = 0
                prob = 0

        train_data = unrolled_data.sample()
        pi_loss, value_loss, ent, learning_rate = actor.train(
                vector=[train_data.vector], front=[train_data.front],
                right=[train_data.right], back=[train_data.back],
                left=[train_data.left], raycast=[train_data.raycast],
                reward=[train_data.reward], done=[train_data.done],
                action_x=[train_data.action_x], action_y=[train_data.action_y],
                action_z=[train_data.action_z], mu_x=[train_data.mu_x],
                mu_y=[train_data.mu_y], mu_z=[train_data.mu_z])

        train_step += 1

        writer.add_scalar('data/pi_loss', pi_loss, train_step)
        writer.add_scalar('data/value_loss', value_loss, train_step)
        writer.add_scalar('data/ent', ent, train_step)
        writer.add_scalar('data/lr', learning_rate, train_step)
        '''

if __name__ == '__main__':
    main()
