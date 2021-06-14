import tensorflow as tf

from baselines.loss.vtrace import from_importance_weights

def split_data(x):
    '''
    x =         [x_(t), x_(t+1), x_(t+2), ..., x_(n)] of shape [None, (n-t+1), ...]
    first_x =   [x_(t), x_(t+1), x_(t+2), ..., x_(n-2)] of shape [None, (n-t-1), ...]
    middle_x =  [x_(t+1), x_(t+2), x_(t+3), ..., x_(n-1)] of shape [None, (n-t-1), ...]
    last_x =    ]x_(t+2), x_(t+3), x_(t+4), ..., x_(n)] of shape [None, (n-t-1), ...]
    '''
    first_x = x[:, :-2]
    middle_x = x[:, 1:-1]
    last_x = x[:, 2:]

    return first_x, middle_x, last_x

def log_probs_from_softmax_combination_action(probs, actions, num_action):

    probs_list = []
    for p, a in zip(probs, actions):
        onehot_action = tf.one_hot(a, num_action)
        selected_softmax = tf.reduce_sum(onehot_action * p, axis=2)
        probs_list.append(selected_softmax)

    combination = tf.ones_like(probs_list[0])
    for p in probs_list:
        combination = combination * p
    return tf.log(combination + 1e-8)

def from_log_probs(behavior_log_prob, target_log_prob, discounts,
                   rewards, values, next_values, clip_rho_threshold=1.0,
                   clip_pg_rho_threshold=1.0):

    log_rhos = target_log_prob - behavior_log_prob
    
    transpose_log_rhos = tf.transpose(log_rhos, perm=[1, 0])
    transpose_discounts = tf.transpose(discounts, perm=[1, 0])
    transpose_rewards = tf.transpose(rewards, perm=[1, 0])
    transpose_values = tf.transpose(values, perm=[1, 0])
    transpose_next_values = tf.transpose(next_values, perm=[1, 0])

    transpose_vs, transpose_clipped_rho = from_importance_weights(
            log_rhos=transpose_log_rhos, discounts=transpose_discounts,
            rewards=transpose_rewards, values=transpose_values,
            bootstrap_value=transpose_next_values[-1],
            clip_rho_threshold=clip_rho_threshold,
            clip_pg_rho_threshold=clip_pg_rho_threshold)

    vs = tf.transpose(transpose_vs, perm=[1, 0])
    clipped_rho = tf.transpose(transpose_clipped_rho, perm=[1, 0])
    return tf.stop_gradient(vs), tf.stop_gradient(clipped_rho)

def compute_policy_gradient_loss(log_prob, advantages):
    policy_gradient_loss_per_timestep = log_prob * advantages
    return -tf.reduce_sum(policy_gradient_loss_per_timestep)

def compute_baseline_loss(vs, value):
    # error = tf.stop_gradient(vs[:, 0]) - value[:, 0]
    error = tf.stop_gradient(vs) - value
    l2_loss = tf.square(error)
    return tf.reduce_sum(l2_loss) * 0.5

def compute_entropy_loss(softmax):
    policy = softmax
    log_policy = tf.log(softmax)
    entropy_per_time_step = -policy * log_policy
    entropy_per_time_step = tf.reduce_sum(entropy_per_time_step, axis=1)
    return -tf.reduce_sum(entropy_per_time_step)
