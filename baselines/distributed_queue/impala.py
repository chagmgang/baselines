import collections
import numpy as np
import tensorflow as tf

class GlobalBuffer:

    def __init__(self, buffer_size):

        self.state = collections.deque(maxlen=int(buffer_size))
        self.reward = collections.deque(maxlen=int(buffer_size))
        self.action = collections.deque(maxlen=int(buffer_size))
        self.done = collections.deque(maxlen=int(buffer_size))
        self.mu = collections.deque(maxlen=int(buffer_size))

    def append(self, state, reward, action, done, mu):

        self.state.append(state)
        self.reward.append(reward)
        self.action.append(action)
        self.done.append(done)
        self.mu.append(mu)

    def __len__(self):
        return len(self.state)

    def sample(self, batch):

        latest_state = self.get_latest_data(self.state, int(batch / 2))
        latest_action = self.get_latest_data(self.action, int(batch / 2))
        latest_reward = self.get_latest_data(self.reward, int(batch / 2))
        latest_done = self.get_latest_data(self.done, int(batch / 2))
        latest_mu = self.get_latest_data(self.mu, int(batch / 2))

        arange = np.arange(len(self.state) - int(batch / 2))
        np.random.shuffle(arange)
        batch_idx = arange[:int(batch / 2)]

        old_state = self.get_old_data(self.state, batch_idx)
        old_action = self.get_old_data(self.action, batch_idx)
        old_reward = self.get_old_data(self.reward, batch_idx)
        old_done = self.get_old_data(self.done, batch_idx)
        old_mu = self.get_old_data(self.mu, batch_idx)

        batch_tuple = collections.namedtuple('batch_tuple',
                ['state', 'reward', 'action', 'done', 'mu'])

        return batch_tuple(
                state=self.extend(old_state, latest_state),
                action=self.extend(old_action, latest_action),
                reward=self.extend(old_reward, latest_reward),
                done=self.extend(old_done, latest_done),
                mu=self.extend(old_mu, latest_mu))

    def extend(self, old, latest):
        l = list()
        l.extend(old)
        l.extend(latest)
        return np.stack(l)


    def get_old_data(self, deque, batch_idx):
        old = [deque[i] for i in batch_idx]
        return np.stack(old)

    def get_latest_data(self, deque, size):
        latest = [deque[-(i+1)] for i in range(size)]
        return np.stack(latest)

class Traj(object):

    def __init__(self, traj):

        self.state = collections.deque(maxlen=traj)
        self.reward = collections.deque(maxlen=traj)
        self.action = collections.deque(maxlen=traj)
        self.done = collections.deque(maxlen=traj)
        self.mu = collections.deque(maxlen=traj)

    def append(self, state, reward, action, done, mu):

        self.state.append(state)
        self.reward.append(reward)
        self.action.append(action)
        self.done.append(done)
        self.mu.append(mu)

class FIFOQueue:

    def __init__(self, traj, input_shape, num_action,
                 queue_size, batch_size, num_actors):

        self.batch_size = batch_size
        self.state = tf.placeholder(tf.float32, [traj, *input_shape])
        self.reward = tf.placeholder(tf.float32, [traj])
        self.action = tf.placeholder(tf.int32, [traj])
        self.done = tf.placeholder(tf.bool, [traj])
        self.mu = tf.placeholder(tf.float32, [traj, num_action])

        self.queue = tf.FIFOQueue(
                queue_size,
                [self.state.dtype,
                 self.reward.dtype,
                 self.action.dtype,
                 self.done.dtype,
                 self.mu.dtype], shared_name='buffer')

        self.queue_size = self.queue.size()

        self.enqueue_ops = []
        for i in range(num_actors):
            self.enqueue_ops.append(
                    self.queue.enqueue(
                        [self.state, self.reward,
                         self.action, self.done,
                         self.mu]))

        self.dequeue = self.queue.dequeue()

    def get_size(self):
        return self.sess.run(self.queue_size)

    def append_to_queue(self, task, state, reward,
                        action, done, mu):

        self.sess.run(
                self.enqueue_ops[task],
                feed_dict={
                    self.state: state, self.reward: reward,
                    self.action: action, self.done: done,
                    self.mu: mu})

    def sample_batch(self):
        batch_tuple = collections.namedtuple('batch_tuple',
            ['state', 'reward', 'action', 'done', 'mu'])

        batch = [self.sess.run(self.dequeue) for i in range(self.batch_size)]

        data = batch_tuple(
                np.stack([i[0] for i in batch]),
                np.stack([i[1] for i in batch]),
                np.stack([i[2] for i in batch]),
                np.stack([i[3] for i in batch]),
                np.stack([i[4] for i in batch]))

        return data

    def set_session(self, sess):
        self.sess = sess
