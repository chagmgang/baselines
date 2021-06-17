import tensorflow as tf
import numpy as np

def cnn(x):
    x = tf.layers.conv2d(
            inputs=x, filters=32, kernel_size=[8, 8],
            strides=[4, 4], padding='VALID', activation=tf.nn.relu)
    x = tf.layers.conv2d(
            inputs=x, filters=64, kernel_size=[4, 4],
            strides=[2, 2], padding='VALID', activation=tf.nn.relu)
    x = tf.layers.conv2d(
            inputs=x, filters=64, kernel_size=[3, 3],
            strides=[1, 1], padding='VALID', activation=tf.nn.relu)
    shape = x.get_shape()
    return tf.layers.flatten(x), [s.value for s in shape]

def fcn(x, hidden_list, num_action, activation):
    for h in hidden_list:
        x = tf.layers.dense(
                inputs=x, units=h, activation=tf.nn.relu)
    return tf.layers.dense(
            inputs=x, units=num_action, activation=activation)

def network(state, num_action):

    x, shape = cnn(state)
    policy = fcn(x, [256, 256], num_action, activation=tf.nn.softmax)
    critic = fcn(x, [256, 256], 1         , activation=None)
    critic = tf.squeeze(critic, axis=1)

    return policy, critic

def build_network(state, traj_state, num_action, traj):

    with tf.variable_scope('impala', reuse=tf.AUTO_REUSE):

        policy, _ = network(state, num_action)

    first_state = traj_state[:, :-2]
    second_state = traj_state[:, 1:-1]
    third_state = traj_state[:, 2:]

    def unrolling(name, state, num_action, traj):
        policy, value = [], []
        for i in range(traj - 2):
            with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
                p, v = network(state[:, i], num_action)
                policy.append(p)
                value.append(v)
        policy = tf.stack(policy, axis=1)
        value = tf.stack(value, axis=1)
        return policy, value

    first_policy, first_value = unrolling(
            name='impala', state=first_state,
            num_action=num_action, traj=traj)
    second_policy, second_value = unrolling(
            name='impala', state=second_state,
            num_action=num_action, traj=traj)
    third_policy, third_value = unrolling(
            name='impala', state=third_state,
            num_action=num_action, traj=traj)

    return policy, first_policy, first_value, second_policy, second_value, \
            third_policy, third_value

def value_network(state, num_action):

    x, shape = cnn(state)
    value = fcn(x, [256, 256], num_action, activation=None)
    return value

def build_value_network(state, next_state, num_action):

    with tf.variable_scope('main'):
        
        main_q_value = value_network(
                state, num_action)

    with tf.variable_scope('main', reuse=tf.AUTO_REUSE):

        main_next_q_value = value_network(
                next_state, num_action)

    with tf.variable_scope('target', reuse=tf.AUTO_REUSE):

        target_next_q_value = value_network(
                next_state, num_action)

    return main_q_value, main_next_q_value, target_next_q_value

if __name__ == '__main__':

    ### test build_value_network function
    state = tf.placeholder(tf.float32, shape=[None, 84, 84, 4])
    next_state = tf.placeholder(tf.float32, shape=[None, 84, 84, 4])
    num_action = 4

    mq_value, mnq_value, tnq_value = build_value_network(state, next_state, num_action)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    traj = np.random.rand(10, 84, 84, 4)
    s = traj[:-1]
    ns = traj[1:]

    a, b, c = sess.run(
            [mq_value, mnq_value, tnq_value],
            feed_dict={
                state: s,
                next_state: ns})

    print(a)
    print(b)
    print(c)

    '''
    ### test build_network function
    state = tf.placeholder(tf.float32, shape=[None, 84, 84, 4])
    traj_state = tf.placeholder(tf.float32, shape=[None, 20, 84, 84, 4])
    num_action = 4
    traj = 20

    a, b, c, d, e, f = build_network(state, traj_state, num_action, traj)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    s_numpy = np.random.rand(1, 84, 84, 4)
    traj_s_numpy = np.random.rand(1, 20, 84, 84, 4)

    result = sess.run(
            [a, b, c, d, e, f], feed_dict={state: s_numpy, traj_state: traj_s_numpy})

    for r in result:
        print(r)
    '''
