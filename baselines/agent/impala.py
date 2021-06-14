import tensorflow as tf
import numpy as np

from baselines import misc
from baselines.loss.vtrace import split_data
from baselines.loss.vtrace import from_softmax
from baselines.loss.vtrace import compute_policy_gradient_loss
from baselines.loss.vtrace import compute_baseline_loss
from baselines.loss.vtrace import compute_entropy_loss
from baselines.model.breakout import build_network

class Agent:

    def __init__(self, trajectory=20, input_shape=[84, 84, 4], num_action=4,
                 discount_factor=0.99, start_learning_rate=0.0006,
                 end_learning_rate=0.0, learning_frame=1000000000,
                 baseline_loss_coef=1.0, ent_coef=0.05,
                 gradient_clip_norm=40.0, reward_clipping='abs_one',
                 model_name='learner', learner_name='learner'):

        with tf.variable_scope(model_name):
            with tf.device('cpu'):
                
                self.s = tf.placeholder(tf.float32, shape=[None, *input_shape])
                self.traj_s = tf.placeholder(tf.float32, shape=[None, trajectory, *input_shape])
                self.a_ph = tf.placeholder(tf.int32, shape=[None, trajectory])
                self.r_ph = tf.placeholder(tf.float32, shape=[None, trajectory])
                self.d_ph = tf.placeholder(tf.bool, shape=[None, trajectory])
                self.b_ph = tf.placeholder(tf.float32, shape=[None, trajectory, num_action])

                if reward_clipping == 'abs_one':
                    self.clipped_r_ph = tf.clip_by_value(self.r_ph, -1.0, 1.0)
                elif reward_clipping == 'soft_asymmetric':
                    squeezed = tf.tanh(self.r_ph / 5.0)
                    self.clipped_r_ph = tf.where(self.r_ph < 0, .3 * squeezed, squeezed) * 5.
                else:
                    self.clipped_r_ph= self.r_ph

                self.discounts = tf.to_float(~self.d_ph) * discount_factor

                self.policy, self.f_p, self.f_v, self.s_p, self.s_v,\
                        self.t_p, self.t_v = build_network(
                                self.s, self.traj_s, num_action, trajectory)

                self.actions = split_data(self.a_ph)
                self.rewards = split_data(self.clipped_r_ph)
                self.disc    = split_data(self.discounts)
                self.behavior_policy = split_data(self.b_ph)

                self.vs, self.clipped_rho = from_softmax(
                        behavior_policy_softmax=self.behavior_policy[0],
                        target_policy_softmax=self.f_p,
                        actions=self.actions[0],
                        discounts=self.disc[0],
                        rewards=self.rewards[0],
                        values=self.f_v,
                        next_values=self.s_v,
                        action_size=num_action)
                

                self.vs_plus_1, _ = from_softmax(
                        behavior_policy_softmax=self.behavior_policy[1],
                        target_policy_softmax=self.s_p,
                        actions=self.actions[1],
                        discounts=self.disc[1],
                        rewards=self.rewards[1],
                        values=self.s_v,
                        next_values=self.t_v,
                        action_size=num_action)

                self.pg_advantage = tf.stop_gradient(
                        self.clipped_rho * \
                                (self.rewards[0] + self.disc[0] * self.vs_plus_1 - self.f_v))

                self.pi_loss = compute_policy_gradient_loss(
                        softmax=self.f_p,
                        actions=self.actions[0],
                        advantages=self.pg_advantage,
                        output_size=num_action)

                self.baseline_loss = compute_baseline_loss(
                        vs=tf.stop_gradient(self.vs),
                        value=self.f_v)

                self.ent = compute_entropy_loss(
                        softmax=self.f_p)

                self.total_loss = self.pi_loss + \
                        self.baseline_loss * baseline_loss_coef + \
                        self.ent * ent_coef

            self.num_env_frames = tf.train.get_or_create_global_step()
            self.learning_rate = tf.train.polynomial_decay(
                    start_learning_rate, self.num_env_frames,
                    learning_frame, end_learning_rate)
            self.optimizer = tf.train.RMSPropOptimizer(
                    self.learning_rate, decay=0.99, momentum=0, epsilon=0.1)
            gradients, variable = zip(*self.optimizer.compute_gradients(self.total_loss))
            gradients, _ = tf.clip_by_global_norm(gradients, gradient_clip_norm)
            self.train_op = self.optimizer.apply_gradients(
                    zip(gradients, variable), global_step=self.num_env_frames)

        self.global_to_session = misc.copy_src_to_dst(learner_name, model_name)
        self.saver = tf.train.Saver()

    def set_session(self, sess):
        self.sess = sess
        self.sess.run(tf.global_variables_initializer())

    def parameter_sync(self):
        self.sess.run(self.global_to_session)

    def save_weights(self, path):
        self.saver.save(self.sess, path)

    def load_weights(self, path):
        self.saver.restore(self.sess, path)

    def get_policy_and_action(self, state):
        policy = self.sess.run(
                self.policy, feed_dict={self.s: [state]})
        policy = policy[0]
        action = np.random.choice(policy.shape[0], p=policy)
        return action, policy, policy[action]

    def train(self, state, reward, action, done, behavior_policy):
        feed_dict = {
                self.traj_s: state,
                self.a_ph: action,
                self.r_ph: reward,
                self.d_ph: done,
                self.b_ph: behavior_policy}

        pi_loss, value_loss, ent, learning_rate, _ = self.sess.run(
                [self.pi_loss, self.baseline_loss, self.ent,
                 self.learning_rate, self.train_op],
                feed_dict=feed_dict)
        
        return pi_loss, value_loss, ent, learning_rate
