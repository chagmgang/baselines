import time
import tensorflow as tf
import numpy as np

from tensorboardX import SummaryWriter

from baselines import misc
from baselines.env.wrappers import make_float_env
from baselines.agent.apex import Agent
from baselines.distributed_queue.apex import Traj
from baselines.distributed_queue.apex import FIFOQueue
from baselines.distributed_queue.apex import Memory

flags = tf.app.flags
FLAGS = tf.app.flags.FLAGS

flags.DEFINE_integer('task', -1, "Task id. Use -1 for local training")
flags.DEFINE_enum('job_name',
                  'learner',
                  ['learner', 'actor'],
                  'Job name. Ignore when task is set to -1')

def main(_):

    num_actors = 8
    server_ip = 'localhost'
    server_port = 8000
    trajectory = 20
    input_shape = [84, 84, 4]
    num_action = 4
    queue_size= 256
    batch_size = 32
    buffer_size = int(1e4)

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
                    input_shape=input_shape,
                    num_action=num_action,
                    queue_size=queue_size,
                    batch_size=batch_size,
                    num_actors=num_actors)

        learner = Agent(
                model_name='learner',
                learner_name='learner')

    with tf.device(local_job_device):

        actor = Agent(
                model_name=f'actor_{FLAGS.task}',
                learner_name='learner')

    sess = tf.Session(server.target)
    learner.set_session(sess)
    queue.set_session(sess)
    if not is_learner:
        actor.set_session(sess)

    if is_learner:

        writer = SummaryWriter('runs/learner')
        buffer_step = 0
        train_step = 0
        replay_buffer = Memory(
                capacity=buffer_size)

        while True:

            size = queue.get_size()
            if size > batch_size:
                sample_data = queue.sample_batch()
                td_error = learner.get_td_error(
                        state=sample_data.state,
                        next_state=sample_data.next_state,
                        action=sample_data.action,
                        reward=sample_data.reward,
                        done=sample_data.done)

                for i in range(len(td_error)):
                    buffer_step += 1
                    replay_buffer.add(
                            td_error[i],
                            [sample_data.state[i],
                             sample_data.next_state[i],
                             sample_data.action[i],
                             sample_data.reward[i],
                             sample_data.done[i]])

            if buffer_step > batch_size * 2:

                train_step += 1
                s = time.time()
                minibatch, idxs, is_weight = replay_buffer.sample(batch_size)
                minibatch = np.array(minibatch)

                state = np.stack(minibatch[:, 0])
                next_state = np.stack(minibatch[:, 1])
                action = np.stack(minibatch[:, 2])
                reward = np.stack(minibatch[:, 3])
                done = np.stack(minibatch[:, 4])

                loss, td_error = learner.train(
                        state, next_state, action,
                        reward, done, is_weight)

                writer.add_scalar('data/loss', loss, train_step)
                writer.add_scalar('data/time', time.time() - s, train_step)

                if train_step % 100 == 0:
                    learner.target_to_main()

                for i in range(len(idxs)):
                    replay_buffer.update(idxs[i], td_error[i])

    else:

        episode = 0
        score = 0
        episode_step = 0
        prob = 0
        lives = 5
        epsilon = 1

        writer = SummaryWriter(f'runs/{FLAGS.task}')
        env = make_float_env("BreakoutDeterministic-v4")
        state = env.reset()

        while True:

            unrolled_data = Traj(traj=trajectory)
            actor.parameter_sync()

            for _ in range(trajectory):

                action, value, _ = actor.get_policy_and_action(
                        state, epsilon)
                next_state, reward, done, info = env.step(action)

                episode_step += 1
                score += reward
                prob += value[action]

                if lives != info['ale.lives']:
                    r = -1
                    d = True
                else:
                    r = reward
                    d = False

                unrolled_data.append(
                        state=state, next_state=next_state,
                        reward=r, done=d,
                        action=action)

                state = next_state
                lives = info['ale.lives']

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
                    lives = 5
                    epsilon = 1 / (episode * 0.05 + 1)

            queue.append_to_queue(
                    task=FLAGS.task,
                    state=unrolled_data.state, next_state=unrolled_data.next_state,
                    action=unrolled_data.action, done=unrolled_data.done,
                    reward=unrolled_data.reward)

if __name__ == '__main__':
    tf.app.run()
