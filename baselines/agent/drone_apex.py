import tensorflow as tf
import numpy as np

from baselines import misc
from baselines.model.drone import build_value_network
from baselines.loss.apex import dqn

class DroneAgent:

    def __init__(self, vector_shape=[10], image_shape=[36, 64, 3], raycast_shape=[16],
                 num_action=27, discount_factor=0.99, gradient_clip_norm=40.0,
                 reward_clipping='abs_one',
                 start_learning_rate=0.0001, end_learning_rate=0.0,
                 learning_frame=100000000000000,
                 model_name='learner', learner_name='learner'):

        with tf.variable_scope(model_name):
            with tf.device('cpu'):

                self.vector = tf.placeholder(tf.float32, shape=[None, *vector_shape])
                self.front = tf.placeholder(tf.float32, shape=[None, *image_shape])
                self.right = tf.placeholder(tf.float32, shape=[None, *image_shape])
                self.back = tf.placeholder(tf.float32, shape=[None, *image_shape])
                self.left = tf.placeholder(tf.float32, shape=[None, *image_shape])
                self.raycast = tf.placeholder(tf.float32, shape=[None, *raycast_shape])

                self.next_vector = tf.placeholder(tf.float32, shape=[None, *vector_shape])
                self.next_front = tf.placeholder(tf.float32, shape=[None, *image_shape])
                self.next_right = tf.placeholder(tf.float32, shape=[None, *image_shape])
                self.next_back = tf.placeholder(tf.float32, shape=[None, *image_shape])
                self.next_left = tf.placeholder(tf.float32, shape=[None, *image_shape])
                self.next_raycast = tf.placeholder(tf.float32, shape=[None, *raycast_shape])

                self.a = tf.placeholder(tf.int32, shape=[None])
                self.r = tf.placeholder(tf.float32, shape=[None])
                self.d = tf.placeholder(tf.bool, shape=[None])
                self.w = tf.placeholder(tf.float32, shape=[None])

                if reward_clipping == 'abs_one':
                    self.clipped_r_ph = tf.clip_by_value(self.r, -1.0, 1.0)
                else:
                    self.clipped_r_ph = self.r

                self.discounts = tf.to_float(~self.d) * discount_factor

                self.main_value, self.main_next_value, self.target_value = build_value_network(
                        self.vector, self.front, self.right, self.back,
                        self.left, self.raycast, self.next_vector,
                        self.next_front, self.next_right, self.next_back,
                        self.next_left, self.next_raycast, num_action)

                self.value, self.target, self.loss = dqn(
                        main_value=self.main_value,
                        main_next_value=self.main_next_value,
                        target_value=self.target_value,
                        reward=self.clipped_r_ph,
                        action=self.a,
                        discounts=self.discounts,
                        weight=self.w,
                        num_action=num_action)

            self.num_env_frames = tf.train.get_or_create_global_step()
            self.learning_rate = tf.train.polynomial_decay(
                    start_learning_rate,
                    self.num_env_frames,
                    learning_frame,
                    end_learning_rate)
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
            gradients, variable = zip(*self.optimizer.compute_gradients(self.loss))
            gradients, _ = tf.clip_by_global_norm(gradients, gradient_clip_norm)
            self.train_op = self.optimizer.apply_gradients(
                    zip(gradients, variable),
                    global_step=self.num_env_frames)

        self.main_target = misc.main_to_target(f'{model_name}/main', f'{model_name}/target')
        self.global_to_session = misc.copy_src_to_dst(learner_name, model_name)
        self.saver = tf.train.Saver()

    def save_weights(self, path):
        self.saver.save(self.sess, path)

    def load_weights(self, path):
        self.saver.restore(self.sess, path)

    def target_to_main(self):
        self.sess.run(self.main_target)

    def parameter_sync(self):
        self.sess.run(self.global_to_session)

    def debug(self):

        batch_size = 3
        vector_shape = [10]
        image_shape = [36, 64, 3]
        raycast_shape = [16]

        vector = np.ones(vector_shape)
        front = np.ones(image_shape)
        right = np.ones(image_shape)
        back = np.ones(image_shape)
        left = np.ones(image_shape)
        raycast = np.ones(raycast_shape)

        print(self.get_policy_and_action(vector, front, right, back, left, raycast, 0.0))

        '''
        vector = np.random.rand(batch_size, *vector_shape)
        front = np.random.rand(batch_size, *image_shape)
        right = np.random.rand(batch_size, *image_shape)
        back = np.random.rand(batch_size, *image_shape)
        left = np.random.rand(batch_size, *image_shape)
        raycast = np.random.rand(batch_size, *raycast_shape)

        action = [0, 1, 2]
        done = [0, 1, 0]
        reward = [0, 1, 0]
        weight = [1, 1, 1]

        self.train(
                vector=vector, front=front, right=right,
                back=back, left=left, raycast=raycast,
                next_vector=vector, next_front=front, next_right=right,
                next_back=back, next_left=left, next_raycast=raycast,
                action=action, done=done, reward=reward, weight=weight)
        '''

    def get_td_error(self, vector, front, right, back, left, raycast,
                     next_vector, next_front, next_right,
                     next_back, next_left, next_raycast,
                     action, reward, done):

        value, target = self.sess.run(
                [self.value, self.target],
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

                    self.a: action,
                    self.d: done,
                    self.r: reward})

        td_error = np.abs(target - value)
        return td_error

    def train(self, vector, front, right, back, left, raycast,
              next_vector, next_front, next_right,
              next_back, next_left, next_raycast,
              action, reward, done, weight):

        loss, value, target, _ = self.sess.run(
                [self.loss, self.value, self.target, self.train_op],
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

                    self.a: action,
                    self.d: done,
                    self.r: reward,
                    self.w: weight})

        td_error = np.abs(target - value)
        return loss, td_error

    def get_policy_and_action(self, vector, front, right, back,
                              left, raycast, epsilon):

        feed_dict = {
                self.vector: [vector],
                self.front: [front],
                self.right: [right],
                self.back: [back],
                self.left: [left],
                self.raycast: [raycast]}

        main_q_value = self.sess.run(
                self.main_value,
                feed_dict=feed_dict)

        main_q_value = main_q_value[0]

        if np.random.rand() > epsilon:
            action = np.argmax(main_q_value, axis=0)
        else:
            shape = main_q_value.shape[0]
            action = np.random.choice(shape)

        return action, main_q_value, main_q_value[action]


    def set_session(self, sess):
        self.sess = sess
        self.sess.run(tf.global_variables_initializer())
