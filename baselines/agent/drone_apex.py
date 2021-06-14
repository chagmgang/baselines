import tensorflow as tf
import numpy as np

from baselines import misc
from baselines.model.drone import build_value_network
from baselines.loss.apex import dqn

class Agent:

    def __init__(self, vector_shape=[10], image_shape=[36, 64, 3], num_action=27,
                 discount_factor=0.99, gradient_clip_norm=40.0, reward_clipping='abs_one',
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

                self.n_vector = tf.placeholder(tf.float32, shape=[None, *vector_shape])
                self.n_front = tf.placeholder(tf.float32, shape=[None, *image_shape])
                self.n_right = tf.placeholder(tf.float32, shape=[None, *image_shape])
                self.n_back = tf.placeholder(tf.float32, shape=[None, *image_shape])
                self.n_left = tf.placeholder(tf.float32, shape=[None, *image_shape])
                self.n_raycast = tf.placeholder(tf.float32, shape=[None, *raycast_shape])

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
                        vector=self.vector, front=self.front, right=self.right, 
                        back=self.back, left=self.left, raycast=self.raycast,
                        next_vector=self.n_vector, next_front=self.n_front,
                        next_right=self.n_right, next_back=self.n_back,
                        next_left=self.n_left, next_raycast=self.n_raycast,
                        num_action=num_action)

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

