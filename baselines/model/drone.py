import tensorflow as tf
import numpy as np

from collections import namedtuple

impala_info = namedtuple(
        'impala_info',
        ['x', 'y', 'z',
         'traj_x', 'traj_y', 'traj_z', 'traj_v'])

def split_trajectory(x):
    return x[:, :-2], x[:, 1:-1], x[:, 2:]

def fcn(x, hidden_list, num_action, activation):
    for h in hidden_list:
        x = tf.layers.dense(
                inputs=x, units=h, activation=tf.nn.relu)
    return tf.layers.dense(
            inputs=x, units=num_action, activation=activation)

def cnn(x, hidden):

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

    x = tf.layers.flatten(x)
    x = fcn(x, [256], num_action=hidden, activation=None)
    return x, [s.value for s in shape]


def feature_concat(vector, front, right, back, left, raycast):

    vector = fcn(vector, [256, 256], num_action=256, activation=None)
    raycast = fcn(raycast, [256, 256], num_action=256, activation=None)
    front, _ = cnn(front, 256)
    right, _ = cnn(right, 256)
    back, _ = cnn(back, 256)
    left, _ = cnn(left, 256)

    summation = vector + raycast + front + right + back + left

    return summation

def network(vector, front, right, back, left, raycast, num_action):

    feature = feature_concat(vector, front, right, back, left, raycast)
    policy = fcn(feature, [256, 256], num_action=num_action, activation=tf.nn.softmax)

    value = fcn(feature, [256, 256], num_action=1, activation=None)
    value = tf.squeeze(value, axis=1)

    return policy, value

def build_network(vector, front, right, back, left, raycast,
                  traj_vector, traj_front, traj_right, traj_back,
                  traj_left, traj_raycast, num_action, trajectory):

    with tf.variable_scope('impala', reuse=tf.AUTO_REUSE):

        policy, _ = network(
                vector, front, right, back, left, raycast, num_action)

    def unrolling(name, traj_vector, traj_front,
                  traj_right, traj_back,
                  traj_left, traj_raycast,
                  num_action, traj):
        policy, value = [], []
        for i in range(traj):
            with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
                p, v = network(traj_vector[:, i],
                               traj_front[:, i],
                               traj_right[:, i],
                               traj_back[:, i],
                               traj_left[:, i],
                               traj_raycast[:, i],
                               num_action)
                policy.append(p)
                value.append(v)
        policy = tf.stack(policy, axis=1)
        value = tf.stack(value, axis=1)
        return policy, value

    traj_policy, traj_value = unrolling(
            name='impala', traj_vector=traj_vector,
            traj_front=traj_front, traj_right=traj_right,
            traj_back=traj_back, traj_left=traj_left,
            traj_raycast=traj_raycast, num_action=num_action,
            traj=trajectory)

    first_policy, second_policy, third_policy = split_trajectory(traj_policy)
    first_value,  second_value,  third_value  = split_trajectory(traj_value)

    return policy, first_policy, first_value, second_policy, second_value, \
            third_policy, third_value

def value_network(vector, front, right, back, left, raycast, num_action):

    feature = feature_concat(
            vector, front, right, back, left, raycast)
    value = fcn(feature, [256, 256], num_action, activation=None)
    return value

def build_value_network(vector, front, right, back, left, raycast,
                        next_vector, next_front, next_right,
                        next_back, next_left, next_raycast,
                        num_action):

    with tf.variable_scope('main'):

        main_q_value = value_network(
                vector, front, right, back, left, raycast, num_action)

    with tf.variable_scope('main', reuse=tf.AUTO_REUSE):

        main_next_q_value = value_network(
                next_vector, next_front, next_right,
                next_back, next_left, next_raycast, num_action)

    with tf.variable_scope('target', reuse=tf.AUTO_REUSE):

        target_next_q_value = value_network(
                next_vector, next_front, next_right,
                next_back, next_left, next_raycast, num_action)

    return main_q_value, main_next_q_value, target_next_q_value

if __name__ == '__main__':

    input_shape = [36, 64, 3]
    vector_shape = [10]
    raycast_shape = [16]
    num_action = 27
    trajectory = 20

    vector = tf.placeholder(tf.float32, shape=[None, *vector_shape])
    front = tf.placeholder(tf.float32, shape=[None, *input_shape])
    right = tf.placeholder(tf.float32, shape=[None, *input_shape])
    back = tf.placeholder(tf.float32, shape=[None, *input_shape])
    left = tf.placeholder(tf.float32, shape=[None, *input_shape])
    raycast = tf.placeholder(tf.float32, shape=[None, *raycast_shape])

    next_vector = tf.placeholder(tf.float32, shape=[None, *vector_shape])
    next_front = tf.placeholder(tf.float32, shape=[None, *input_shape])
    next_right = tf.placeholder(tf.float32, shape=[None, *input_shape])
    next_back = tf.placeholder(tf.float32, shape=[None, *input_shape])
    next_left = tf.placeholder(tf.float32, shape=[None, *input_shape])
    next_raycast = tf.placeholder(tf.float32, shape=[None, *raycast_shape])

    build_value_network(
            vector, front, right, back, left, raycast,
            next_vector, next_front, next_right,
            next_back, next_left, next_raycast, num_action)

    '''
    batch_size = 1

    vector = tf.placeholder(tf.float32, shape=[None, *vector_shape])
    front = tf.placeholder(tf.float32, shape=[None, *input_shape])
    right = tf.placeholder(tf.float32, shape=[None, *input_shape])
    back = tf.placeholder(tf.float32, shape=[None, *input_shape])
    left = tf.placeholder(tf.float32, shape=[None, *input_shape])
    raycast = tf.placeholder(tf.float32, shape=[None, *raycast_shape])

    network(vector, front, right, back, left, raycast, num_action)

    traj_vector = tf.placeholder(tf.float32, shape=[None, trajectory, *vector_shape])
    traj_front = tf.placeholder(tf.float32, shape=[None, trajectory, *input_shape])
    traj_right = tf.placeholder(tf.float32, shape=[None, trajectory, *input_shape])
    traj_back = tf.placeholder(tf.float32, shape=[None, trajectory, *input_shape])
    traj_left = tf.placeholder(tf.float32, shape=[None, trajectory, *input_shape])
    traj_raycast = tf.placeholder(tf.float32, shape=[None, trajectory, *raycast_shape])

    xx = build_network(
            vector, front, right, back ,left, raycast,
            traj_vector, traj_front, traj_right, traj_back, traj_left, traj_raycast,
            num_action, trajectory)

    for x in xx:
        print(x)
    print(vector)
    print(front)
    print(right)
    print(back)
    print(left)
    print(raycast)

    impala_info = build_network(
            vector, front, right, back, left, raycast,
            traj_vector, traj_front, traj_right, traj_back,
            traj_left, traj_raycast, num_action, trajectory)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    traj_vector_np = np.random.rand(1, trajectory, *vector_shape)
    traj_front_np  = np.random.rand(1, trajectory, *input_shape)
    traj_right_np  = np.random.rand(1, trajectory, *input_shape)
    traj_back_np   = np.random.rand(1, trajectory, *input_shape)
    traj_left_np   = np.random.rand(1, trajectory, *input_shape)
    traj_raycast_np= np.random.rand(1, trajectory, *raycast_shape)

    result = sess.run(
            [impala_info.f_x, impala_info.s_x, impala_info.t_x, impala_info.x],
            feed_dict={
                traj_vector: traj_vector_np,
                traj_front: traj_front_np,
                traj_right: traj_right_np,
                traj_back: traj_back_np,
                traj_left: traj_left_np,
                traj_raycast: traj_raycast_np,
                
                vector: traj_vector_np[:, 0],
                front: traj_front_np[:, 0],
                right: traj_right_np[:, 0],
                back: traj_back_np[:, 0],
                left: traj_left_np[:, 0],
                raycast: traj_raycast_np[:, 0]})


    for r in result:
        print(r)
    '''
