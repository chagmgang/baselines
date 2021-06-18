from baselines.env.drone import Drone
from baselines.agent.drone_apex import DroneAgent
from baselines.distributed_queue.drone_apex import Traj
from baselines.distributed_queue.drone_apex import FIFOQueue
from baselines.distributed_queue.drone_apex import Memory
from baselines.misc import convert_action

from tensorboardX import SummaryWriter

import tensorflow as tf
import numpy as np

import time

flags = tf.app.flags
FLAGS = tf.app.flags.FLAGS

flags.DEFINE_integer('task', -1, "Task id, Use -1 for local training")
flags.DEFINE_enum('job_name',
                  'learner',
                  ['learner', 'actor'],
                  'Job name, Ignore when task is set to -1')

def main(_):

    num_actors = 3
    server_ip = 'localhost'
    server_port = 8000
    trajectory = 20
    queue_size = 256
    batch_size = 32
    buffer_size = 1e5

    local_job_device = f'/job:{FLAGS.job_name}/task:{FLAGS.task}'
    shared_job_device = '/job:learner/task:0'
    is_learner = FLAGS.job_name == 'learner'

    cluster = tf.train.ClusterSpec({
        'actor': ['{}:{}'.format(server_ip, server_port+i+1) for i in range(num_actors)],
        'learner': ['{}:{}'.format(server_ip, server_port)]})

    server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task)

    with tf.device(shared_job_device):

        with tf.device('/cpu'):
            queue = FIFOQueue(
                    traj=trajectory,
                    batch_size=batch_size,
                    queue_size=queue_size,
                    num_actors=num_actors)

        learner = DroneAgent(
                model_name='learner',
                learner_name='learner')

    with tf.device(local_job_device):

        actor = DroneAgent(
                model_name=f'actor_{FLAGS.task}',
                learner_name='learner')

    sess = tf.Session(server.target)
    learner.set_session(sess)
    queue.set_session(sess)
    if not is_learner:
        actor.set_session(sess)

    if is_learner:

        learner.target_to_main()
        replay_buffer = Memory(capacity=int(buffer_size))
        buffer_step = 0
        train_step = 0

        writer = SummaryWriter('runs/learner')

        while True:

            size = queue.get_size()
            if size > 3 * batch_size:
                sample_data = queue.sample_batch()
                
                for i in range(batch_size):
                    td_error = learner.get_td_error(
                            vector=sample_data.vector[i],
                            front=sample_data.front[i],
                            right=sample_data.right[i],
                            back=sample_data.back[i],
                            left=sample_data.left[i],
                            raycast=sample_data.raycast[i],
                            next_vector=sample_data.next_vector[i],
                            next_front=sample_data.next_front[i],
                            next_right=sample_data.next_right[i],
                            next_back=sample_data.next_back[i],
                            next_left=sample_data.next_left[i],
                            next_raycast=sample_data.next_raycast[i],
                            action=sample_data.action[i],
                            reward=sample_data.reward[i],
                            done=sample_data.done[i])

                    for j in range(len(td_error)):
                        buffer_step += 1
                        replay_buffer.add(
                                td_error[j],
                                [sample_data.vector[i, j],
                                 sample_data.front[i, j],
                                 sample_data.right[i, j],
                                 sample_data.back[i, j],
                                 sample_data.left[i, j],
                                 sample_data.raycast[i, j],
                                 sample_data.next_vector[i, j],
                                 sample_data.next_front[i, j],
                                 sample_data.next_right[i, j],
                                 sample_data.next_back[i, j],
                                 sample_data.next_left[i, j],
                                 sample_data.next_raycast[i, j],
                                 sample_data.action[i, j],
                                 sample_data.reward[i, j],
                                 sample_data.done[i, j]])

                        if buffer_step > buffer_size:
                            buffer_step = buffer_size

            if buffer_step > 3 * batch_size:

                train_step += 1

                s = time.time()

                minibatch, idxs, is_weight = replay_buffer.sample(batch_size)
                minibatch = np.array(minibatch)

                vector = np.stack(minibatch[:, 0])
                front = np.stack(minibatch[:, 1])
                right = np.stack(minibatch[:, 2])
                back = np.stack(minibatch[:, 3])
                left = np.stack(minibatch[:, 4])
                raycast = np.stack(minibatch[:, 5])
                next_vector = np.stack(minibatch[:, 6])
                next_front = np.stack(minibatch[:, 7])
                next_right = np.stack(minibatch[:, 8])
                next_back = np.stack(minibatch[:, 9])
                next_left = np.stack(minibatch[:, 10])
                next_raycast = np.stack(minibatch[:, 11])
                action = np.stack(minibatch[:, 12])
                reward = np.stack(minibatch[:, 13])
                done = np.stack(minibatch[:, 14])

                loss, td_error = learner.train(
                        vector=vector, front=front,
                        right=right, back=back,
                        left=left, raycast=raycast,

                        next_vector=next_vector, next_front=next_front,
                        next_right=next_right, next_back=next_back,
                        next_left=next_left, next_raycast=next_raycast,

                        action=action, reward=reward, done=done, weight=is_weight)

                writer.add_scalar('data/buffer_size', buffer_step, train_step)
                writer.add_scalar('data/loss', loss, train_step)
                writer.add_scalar('data/time', time.time() - s, train_step)

                if train_step % 100 == 0:
                    learner.target_to_main()
                
                if train_step % 1000 == 0:
                    learner.save_weights('saved/model', step=train_step)

                for i in range(len(idxs)):
                    replay_buffer.update(idxs[i], td_error[i])

                print(f'train : {train_step}')

    else:

        episode = 0
        score = 0
        episode_step = 0
        prob = 0
        epsilon = 1

        writer = SummaryWriter(f'runs/{FLAGS.task}')
        env = Drone(
                time_scale=0.05,
                port=11000+FLAGS.task,
                filename='/Users/chageumgang/Desktop/baselines/mac.app')

        state = env.reset()

        while True:

            unrolled_data = Traj(traj=trajectory)
            actor.parameter_sync()

            for _ in range(trajectory):

                action, behavior_policy, _ = actor.get_policy_and_action(
                        vector=state.vector, front=state.front,
                        right=state.right, back=state.back,
                        left=state.left, raycast=state.raycast,
                        epsilon=epsilon)

                next_state, reward, done = env.step(convert_action(action))

                episode_step += 1
                score += reward
                prob += behavior_policy[action]

                unrolled_data.append(
                        vector=state.vector, front=state.front,
                        right=state.right, back=state.back,
                        left=state.left, raycast=state.raycast,

                        next_vector=next_state.vector, next_front=next_state.front,
                        next_right=next_state.right, next_back=next_state.back,
                        next_left=next_state.left, next_raycast=next_state.raycast,

                        reward=reward, done=done, action=action)

                state = next_state

                if done:

                    print(episode, score)
                    writer.add_scalar('data/score', score, episode)
                    writer.add_scalar('data/prob', prob / episode_step, episode)
                    writer.add_scalar('data/episode_step', episode_step, episode)
                    writer.add_scalar('data/epsilon', epsilon, episode)

                    state = env.reset()
                    episode += 1
                    score = 0
                    episode_step = 0
                    prob = 0
                    epsilon = 1 / (episode * 0.05 + 1)

            train_data = unrolled_data.sample()
            queue.append_to_queue(
                    task=FLAGS.task,
                    vector=train_data.vector, front=train_data.front,
                    right=train_data.right, back=train_data.back,
                    left=train_data.left, raycast=train_data.raycast,

                    next_vector=train_data.next_vector, next_front=train_data.next_front,
                    next_right=train_data.next_right, next_back=train_data.next_back,
                    next_left=train_data.next_left, next_raycast=train_data.next_raycast,

                    done=train_data.done, action=train_data.action, reward=train_data.reward)



if __name__ == '__main__':
    tf.app.run()
