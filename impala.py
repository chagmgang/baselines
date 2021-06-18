from baselines.env.drone import Drone
from baselines.agent.drone_impala import DroneAgent
from baselines.distributed_queue.drone_impala import Traj
from baselines.distributed_queue.drone_impala import FIFOQueue
from baselines.distributed_queue.drone_impala import GlobalBuffer
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

    num_actors = 1
    server_ip = 'localhost'
    server_port = 8000
    trajectory = 20
    queue_size = 256
    batch_size = 32
    buffer_size = 1e4

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
                trajectory=trajectory,
                model_name='learner',
                learner_name='learner')

    with tf.device(local_job_device):

        actor = DroneAgent(
                trajectory=trajectory,
                model_name=f'actor_{FLAGS.task}',
                learner_name='learner')

    sess = tf.Session(server.target)
    learner.set_session(sess)
    queue.set_session(sess)
    if not is_learner:
        actor.set_session(sess)

    if is_learner:

        global_buffer = GlobalBuffer(
                buffer_size=buffer_size)

        train_step = 0
        writer = SummaryWriter('runs/learner')

        while True:

            size = queue.get_size()
            if size > batch_size:
                sample_data = queue.sample_batch()
                for i in range(len(sample_data.vector)):
                    global_buffer.append(
                            vector=sample_data.vector[i],
                            front=sample_data.front[i],
                            right=sample_data.right[i],
                            back=sample_data.back[i],
                            left=sample_data.left[i],
                            raycast=sample_data.raycast[i],
                            reward=sample_data.reward[i],
                            done=sample_data.done[i],
                            action=sample_data.action[i],
                            mu=sample_data.mu[i])

            if len(global_buffer) > 3 * batch_size:

                train_step += 1
                train_data = global_buffer.sample(batch_size)
                s = time.time()
                pi_loss, value_loss, ent, lr = learner.train(
                        vector=train_data.vector,
                        front=train_data.front,
                        right=train_data.right,
                        left=train_data.left,
                        back=train_data.back,
                        raycast=train_data.raycast,
                        reward=train_data.reward,
                        done=train_data.done,
                        action=train_data.action,
                        behavior_policy=train_data.mu)

                if train_step % 1000 == 0:
                    learner.save_weights('saved/model', step=train_step)
                print(f'train : {train_step}')

                writer.add_scalar('data/time', time.time() - s, train_step)
                writer.add_scalar('data/pi_loss', pi_loss, train_step)
                writer.add_scalar('data/value_loss', value_loss, train_step)
                writer.add_scalar('data/ent', ent, train_step)
                writer.add_scalar('data/lr', lr, train_step)
                writer.add_scalar('data/buffer', len(global_buffer), train_step)

    else:

        episode = 0
        score = 0
        episode_step = 0
        prob = 0

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
                        left=state.left, raycast=state.raycast)

                next_state, reward, done = env.step(convert_action(action))

                episode_step += 1
                score += reward
                prob += behavior_policy[action]

                unrolled_data.append(
                        vector=state.vector, front=state.front,
                        right=state.right, back=state.back,
                        left=state.left, raycast=state.raycast,
                        reward=reward, done=done,
                        action=action,
                        mu=behavior_policy)

                state = next_state

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
            queue.append_to_queue(
                    task=FLAGS.task, vector=train_data.vector,
                    front=train_data.front, right=train_data.right,
                    back=train_data.back, left=train_data.left,
                    raycast=train_data.raycast, reward=train_data.reward,
                    done=train_data.done, action=train_data.action,
                    mu=train_data.mu)

if __name__ == '__main__':
    tf.app.run()
