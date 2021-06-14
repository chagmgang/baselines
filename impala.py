import time
import tensorflow as tf
import numpy as np

from tensorboardX import SummaryWriter

from baselines import misc
from baselines.env.wrappers import make_float_env
from baselines.agent.impala import Agent
from baselines.distributed_queue.impala import Traj
from baselines.distributed_queue.impala import FIFOQueue
from baselines.distributed_queue.impala import GlobalBuffer

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
    batch_size = 4
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
                    input_shape=input_shape,
                    num_action=num_action,
                    queue_size=queue_size,
                    batch_size=batch_size,
                    num_actors=num_actors)


        learner = Agent(
                trajectory=trajectory,
                model_name='learner',
                learner_name='learner')

    with tf.device(local_job_device):

        actor = Agent(
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
        writer = SummaryWriter('runs/learner')

        train_step = 0

        while True:

            size = queue.get_size()
            if size > batch_size:
                sample_data = queue.sample_batch()
                for i in range(len(sample_data.state)):
                    global_buffer.append(
                        state=sample_data.state[i],
                        reward=sample_data.reward[i],
                        action=sample_data.action[i],
                        done=sample_data.done[i],
                        mu=sample_data.mu[i])

            if len(global_buffer) > 3 * batch_size:
                
                train_step += 1
                train_data = global_buffer.sample(batch_size)

                s = time.time()
                pi_loss, value_loss, ent, lr = learner.train(
                        state=train_data.state,
                        reward=train_data.reward,
                        action=train_data.action,
                        done=train_data.done,
                        behavior_policy=train_data.mu)

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
        lives = 5

        writer = SummaryWriter(f'runs/{FLAGS.task}')
        env = make_float_env("BreakoutDeterministic-v4")
        state = env.reset()

        while True:

            unrolled_data = Traj(traj=trajectory)
            actor.parameter_sync()

            for _ in range(trajectory):

                action, behavior_policy, _ = actor.get_policy_and_action(state)
                next_state, reward, done, info = env.step(action)

                episode_step += 1
                score += reward
                prob += behavior_policy[action]

                if lives != info['ale.lives']:
                    r = -1
                    d = True
                else:
                    r = reward
                    d = False

                unrolled_data.append(
                    state=state, reward=r,
                    action=action, done=d,
                    mu=behavior_policy)

                state = next_state
                lives = info['ale.lives']

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
                    lives = 5

            queue.append_to_queue(
                    task=FLAGS.task,
                    state=unrolled_data.state, reward=unrolled_data.reward,
                    action=unrolled_data.action, done=unrolled_data.done,
                    mu=unrolled_data.mu)



if __name__ == '__main__':
    tf.app.run()
