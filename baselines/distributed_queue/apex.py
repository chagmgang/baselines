import collections
import numpy as np
import tensorflow as tf
import random

class SumTree:
    write = 0
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.n_entries = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, p, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, p)
        self.write += 1
        if self.write >= self.capacity:
            self.write = 0
        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        return (idx, self.tree[idx], self.data[dataIdx])

class Memory(object):
    e = 0.001
    a = 0.6
    beta = 0.4
    beta_increment_per_sampling = 0.001

    def __init__(self, capacity):
        self.tree = SumTree(capacity)
        self.capacity = capacity

    def reset(self):
        self.tree = SumTree(self.capacity)

    def _getPriority(self, error):
        return (error + self.e) ** self.a

    def add(self, error, sample):
        p = self._getPriority(error)
        self.tree.add(p, sample)

    def sample(self, n):
        batch = []
        idxs = []
        segment = self.tree.total() / n
        priorities = []
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        for i in range(n):
            a, b = segment * i, segment * (i+1)
            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)

        sampling_probabilities = priorities / self.tree.total()
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()

        return batch, idxs, is_weight

    def update(self, idx, error):
        p = self._getPriority(error)
        self.tree.update(idx, p)


class Traj(object):

    def __init__(self, traj):

        self.state = collections.deque(maxlen=traj)
        self.next_state = collections.deque(maxlen=traj)
        self.reward = collections.deque(maxlen=traj)
        self.action = collections.deque(maxlen=traj)
        self.done = collections.deque(maxlen=traj)

    def append(self, state, next_state,
               reward, action, done):

        self.state.append(state)
        self.next_state.append(next_state)
        self.reward.append(reward)
        self.action.append(action)
        self.done.append(done)

class FIFOQueue:

    def __init__(self, traj, input_shape, num_action,
                 queue_size, batch_size, num_actors):

        self.batch_size = batch_size
        self.state = tf.placeholder(tf.float32, [traj, *input_shape])
        self.next_state = tf.placeholder(tf.float32, [traj, *input_shape])
        self.reward = tf.placeholder(tf.float32, [traj])
        self.action = tf.placeholder(tf.int32, [traj])
        self.done = tf.placeholder(tf.bool, [traj])

        self.queue = tf.FIFOQueue(
                queue_size,
                [self.state.dtype,
                 self.next_state.dtype,
                 self.reward.dtype,
                 self.action.dtype,
                 self.done.dtype], shared_name='buffer')

        self.queue_size = self.queue.size()

        self.enqueue_ops = []
        for i in range(num_actors):
            self.enqueue_ops.append(
                    self.queue.enqueue(
                        [self.state, self.next_state,
                         self.reward, self.action, self.done]))

        self.dequeue = self.queue.dequeue()

    def get_size(self):
        return self.sess.run(self.queue_size)

    def append_to_queue(self, task, state, next_state,
                        reward, action, done):

        self.sess.run(
                self.enqueue_ops[task],
                feed_dict={
                    self.state: state, self.reward: reward,
                    self.action: action, self.done: done,
                    self.next_state: next_state})

    def sample_batch(self):
        batch_tuple = collections.namedtuple('batch_tuple',
            ['state', 'next_state', 'reward', 'action', 'done'])

        batch = [self.sess.run(self.dequeue) for i in range(1)]

        data = batch_tuple(
                np.stack([i[0] for i in batch])[0],
                np.stack([i[1] for i in batch])[0],
                np.stack([i[2] for i in batch])[0],
                np.stack([i[3] for i in batch])[0],
                np.stack([i[4] for i in batch])[0])

        return data

    def set_session(self, sess):
        self.sess = sess
