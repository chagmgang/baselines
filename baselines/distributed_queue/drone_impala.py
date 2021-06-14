import collections
import numpy as np
import tensorflow as tf

traj_tuple = collections.namedtuple('traj_tuple',
        ['vector', 'front', 'right',
         'back', 'left', 'raycast',
         'reward', 'done',
         'action', 'mu'])

class GlobalBuffer:

    def __init__(self, buffer_size):

        buffer_size = int(buffer_size)
        self.vector = collections.deque(maxlen=buffer_size)
        self.front = collections.deque(maxlen=buffer_size)
        self.right = collections.deque(maxlen=buffer_size)
        self.back = collections.deque(maxlen=buffer_size)
        self.left = collections.deque(maxlen=buffer_size)
        self.raycast = collections.deque(maxlen=buffer_size)

        self.reward = collections.deque(maxlen=buffer_size)
        self.done = collections.deque(maxlen=buffer_size)
        self.action = collections.deque(maxlen=buffer_size)
        self.mu = collections.deque(maxlen=buffer_size)

    def append(self, vector, front, right, back,
               left, raycast, reward, done,
               action, mu):

        self.vector.append(vector)
        self.front.append(front)
        self.right.append(right)
        self.back.append(back)
        self.left.append(left)
        self.raycast.append(raycast)
        self.reward.append(reward)
        self.done.append(done)
        self.action.append(action)
        self.mu.append(mu)

    def sample(self, batch):

        latest_vector = self.get_latest_data(self.vector, int(batch / 2))
        latest_front = self.get_latest_data(self.front, int(batch / 2))
        latest_right = self.get_latest_data(self.right, int(batch / 2))
        latest_back = self.get_latest_data(self.back, int(batch / 2))
        latest_left = self.get_latest_data(self.left, int(batch / 2))
        latest_raycast = self.get_latest_data(self.raycast, int(batch / 2))

        latest_reward = self.get_latest_data(self.reward, int(batch / 2))
        latest_done = self.get_latest_data(self.done, int(batch / 2))
        latest_action = self.get_latest_data(self.action, int(batch / 2))
        latest_mu = self.get_latest_data(self.mu, int(batch / 2))

        arange = np.arange(len(self.vector) - int(batch / 2))
        np.random.shuffle(arange)
        batch_idx = arange[:int(batch / 2)]

        old_vector = self.get_old_data(self.vector, batch_idx)
        old_front = self.get_old_data(self.front, batch_idx)
        old_right = self.get_old_data(self.right, batch_idx)
        old_back = self.get_old_data(self.back, batch_idx)
        old_left = self.get_old_data(self.left, batch_idx)
        old_raycast = self.get_old_data(self.raycast, batch_idx)

        old_reward = self.get_old_data(self.reward, batch_idx)
        old_done = self.get_old_data(self.done, batch_idx)
        old_action = self.get_old_data(self.action, batch_idx)
        old_mu = self.get_old_data(self.mu, batch_idx)

        return traj_tuple(
                vector=self.extend(old_vector, latest_vector),
                front=self.extend(old_front, latest_front),
                right=self.extend(old_right, latest_right),
                back=self.extend(old_back, latest_back),
                left=self.extend(old_left, latest_left),
                raycast=self.extend(old_raycast, latest_raycast),
                reward=self.extend(old_reward, latest_reward),
                done=self.extend(old_done, latest_done),
                action=self.extend(old_action, latest_action),
                mu=self.extend(old_mu, latest_mu))

    def __len__(self):
        return len(self.vector)

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

        self.reward = tf.placeholder(tf.float32, shape=[traj])
        self.done = tf.placeholder(tf.bool, shape=[traj])
        self.action = tf.placeholder(tf.int32, shape=[traj])
        self.mu = tf.placeholder(tf.float32, shape=[traj, num_action])
        
        self.queue = tf.FIFOQueue(
                queue_size,
                [self.vector.dtype,
                 self.front.dtype,
                 self.right.dtype,
                 self.back.dtype,
                 self.left.dtype,
                 self.raycast.dtype,

                 self.reward.dtype,
                 self.done.dtype,
                 self.action.dtype,
                 self.mu.dtype], shared_name='buffer')

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

                          self.reward,
                          self.done,
                          self.action,
                          self.mu]))

        self.dequeue = self.queue.dequeue()

    def sample_batch(self):
        batch = [self.sess.run(self.dequeue) for i in range(self.batch_size)]

        batch_tuple = []
        for i in range(10):
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
                        back, left, raycast, reward, done,
                        action, mu):

        self.sess.run(
                self.enqueue_ops[task],
                feed_dict={
                    self.vector: vector,
                    self.front: front,
                    self.right: right,
                    self.back: back,
                    self.left: left,
                    self.raycast: raycast,
                    self.reward: reward,
                    self.done: done,
                    self.action: action,
                    self.mu: mu})

class Traj(object):

    def __init__(self, traj):

        self.vector = collections.deque(maxlen=traj)
        self.front = collections.deque(maxlen=traj)
        self.right = collections.deque(maxlen=traj)
        self.back = collections.deque(maxlen=traj)
        self.left = collections.deque(maxlen=traj)
        self.raycast = collections.deque(maxlen=traj)
        
        self.reward = collections.deque(maxlen=traj)
        self.done = collections.deque(maxlen=traj)
        self.action = collections.deque(maxlen=traj)
        self.mu = collections.deque(maxlen=traj)

    def append(self, vector, front, right, back, left, raycast,
               reward, done, action, mu):

        self.vector.append(vector)
        self.front.append(front)
        self.right.append(right)
        self.back.append(back)
        self.left.append(left)
        self.raycast.append(raycast)
        self.reward.append(reward)
        self.done.append(done)
        self.action.append(action)
        self.mu.append(mu)

    def sample(self):

        return traj_tuple(
                vector=np.stack(self.vector), front=np.stack(self.front),
                right=np.stack(self.right), back=np.stack(self.back),
                left=np.stack(self.left), raycast=np.stack(self.raycast),
                reward=np.stack(self.reward), done=np.stack(self.done),
                action=np.stack(self.action), mu=np.stack(self.mu))
