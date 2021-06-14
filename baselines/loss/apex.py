import tensorflow as tf

def get_value(value, action, num_action):

    onehot_action = tf.one_hot(action, num_action)
    state_action_value = value * onehot_action
    state_action_value = tf.reduce_sum(state_action_value, axis=1)
    return state_action_value


def dqn(main_value, main_next_value, target_value,
        reward, action, discounts, weight, num_action):

    main_value = get_value(main_value, action, num_action) 
    next_action = tf.argmax(main_next_value, axis=1)
    next_value = get_value(target_value, next_action, num_action)

    target = tf.stop_gradient(
            reward + discounts * next_value)

    td_error = (target - main_value) ** 2
    weighted_td_error = td_error * weight
    
    loss = tf.reduce_mean(weighted_td_error)
    return main_value, target, loss
