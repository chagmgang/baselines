import random
import collections
import numpy as np
import tensorflow as tf

traj_tuple = collections.namedtuple('traj_tuple',
        ['vector', 'front', 'right',
         'back', 'left', 'raycast',
         'next_vector', 'next_front', 'next_right',
         'next_back', 'next_left', 'next_raycast',
         'reward', 'done', 'action'])

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

class FIFOQueue:

    def __init__(self, traj=20, vector_shape=[10],
                 image_shape=[36, 64, 3], raycast_shape=[16],
                 num_action=27, queue_size=256, batch_size=32,
                 num_actors=8):

        self.batch_size = batch_size
        
        self.vector = tf.placeholder(tf.float32, [traj, *vector_shape])
        self.front = tf.placeholder(tf.float32, [traj, *image_shape])
        self.right = tf.placeholder(tf.float32, [traj, *image_shape])
        self.back = tf.placeholder(tf.float32, [traj, *image_shape])
        self.left = tf.placeholder(tf.float32, [traj, *image_shape])
        self.raycast = tf.placeholder(tf.float32, [traj, *raycast_shape])

        self.next_vector = tf.placeholder(tf.float32, [traj, *vector_shape])
        self.next_front = tf.placeholder(tf.float32, [traj, *image_shape])
        self.next_right = tf.placeholder(tf.float32, [traj, *image_shape])
        self.next_back = tf.placeholder(tf.float32, [traj, *image_shape])
        self.next_left = tf.placeholder(tf.float32, [traj, *image_shape])
        self.next_raycast = tf.placeholder(tf.float32, [traj, *raycast_shape])

        self.reward = tf.placeholder(tf.float32, shape=[traj])
        self.done = tf.placeholder(tf.bool, shape=[traj])
        self.action = tf.placeholder(tf.int32, shape=[traj])
        
        self.queue = tf.FIFOQueue(
                queue_size,
                [self.vector.dtype,
                 self.front.dtype,
                 self.right.dtype,
                 self.back.dtype,
                 self.left.dtype,
                 self.raycast.dtype,

                 self.next_vector.dtype,
                 self.next_front.dtype,
                 self.next_right.dtype,
                 self.next_back.dtype,
                 self.next_left.dtype,
                 self.next_raycast.dtype,

                 self.reward.dtype,
                 self.done.dtype,
                 self.action.dtype], shared_name='buffer')

        self.queue_size = self.queue.size()

        self.enqueue_ops = []
        for i in range(num_actors):
            self.enqueue_ops.append(
                    self.queue.enqueue(
                         [self.vector,
                          self.front,
                          self.right,
                          self.back,
                          self.left,
                          self.raycast,

                          self.next_vector,
                          self.next_front,
                          self.next_right,
                          self.next_back,
                          self.next_left,
                          self.next_raycast,

                          self.reward,
                          self.done,
                          self.action]))

        self.dequeue = self.queue.dequeue()

    def sample_batch(self):
        batch = [self.sess.run(self.dequeue) for i in range(self.batch_size)]

        batch_tuple = []
        for i in range(15):
            tup = [j[i] for j in batch]
            batch_tuple.append(np.array(tup))

        data = traj_tuple(
                *batch_tuple)

        return data

    def set_session(self, sess):
        self.sess = sess

    def get_size(self):
        return self.sess.run(self.queue_size)

    def append_to_queue(self, task, vector, front, right,
                        back, left, raycast,

                        next_vector, next_front, next_right,
                        next_back, next_left, next_raycast,

                        reward, done,
                        action):

        self.sess.run(
                self.enqueue_ops[task],
                feed_dict={
                    self.vector: vector,
                    self.front: front,
                    self.right: right,
                    self.back: back,
                    self.left: left,
                    self.raycast: raycast,

                    self.next_vector: next_vector,
                    self.next_front: next_front,
                    self.next_right: next_right,
                    self.next_back: next_back,
                    self.next_left: next_left,
                    self.next_raycast: next_raycast,

                    self.reward: reward,
                    self.done: done,
                    self.action: action})

class Traj(object):

    def __init__(self, traj):

        self.vector = collections.deque(maxlen=traj)
        self.front = collections.deque(maxlen=traj)
        self.right = collections.deque(maxlen=traj)
        self.back = collections.deque(maxlen=traj)
        self.left = collections.deque(maxlen=traj)
        self.raycast = collections.deque(maxlen=traj)

        self.next_vector = collections.deque(maxlen=traj)
        self.next_front = collections.deque(maxlen=traj)
        self.next_right = collections.deque(maxlen=traj)
        self.next_back = collections.deque(maxlen=traj)
        self.next_left = collections.deque(maxlen=traj)
        self.next_raycast = collections.deque(maxlen=traj)

        self.reward = collections.deque(maxlen=traj)
        self.done = collections.deque(maxlen=traj)
        self.action = collections.deque(maxlen=traj)

    def append(self, vector, front, right, back, left, raycast,
               next_vector, next_front, next_right, next_back,
               next_left, next_raycast, reward, done, action):

        self.vector.append(vector)
        self.front.append(front)
        self.right.append(right)
        self.back.append(back)
        self.left.append(left)
        self.raycast.append(raycast)

        self.next_vector.append(next_vector)
        self.next_front.append(next_front)
        self.next_right.append(next_right)
        self.next_back.append(next_back)
        self.next_left.append(next_left)
        self.next_raycast.append(next_raycast)

        self.reward.append(reward)
        self.done.append(done)
        self.action.append(action)

    def sample(self):

        return traj_tuple(
                vector=np.stack(self.vector), front=np.stack(self.front),
                right=np.stack(self.right), back=np.stack(self.back),
                left=np.stack(self.left), raycast=np.stack(self.raycast),

                next_vector=np.stack(self.next_vector), next_front=np.stack(self.next_front),
                next_right=np.stack(self.next_right), next_back=np.stack(self.next_back),
                next_left=np.stack(self.next_left), next_raycast=np.stack(self.next_raycast),

                reward=np.stack(self.reward), done=np.stack(self.done),
                action=np.stack(self.action))
