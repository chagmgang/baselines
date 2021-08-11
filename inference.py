from baselines.env.drone import Drone
from baselines.agent.drone_impala import DroneAgent

import tensorflow as tf
import numpy as np

import time

from baselines.misc import convert_action

def main():

    trajectory = 20
    vector_shape = [14]
    image_shape = [36, 64, 3]
    raycast_shape = [22]

    actor = DroneAgent(
            trajectory=trajectory,
            vector_shape=vector_shape,
            image_shape=image_shape,
            raycast_shape=raycast_shape,
            model_name='learner',
            learner_name='learner',
            )

    sess = tf.Session()
    actor.set_session(sess)
    actor.load_weights('saved_impala/model-123000')

    episode = 0
    score = 0
    
    env = Drone(
            time_scale=0.05,
            port=11199,
            filename='/Users/chageumgang/Desktop/baselines/mac_2.app')

    state = env.reset()

    while True:

        _, behavior_policy, _ = actor.get_policy_and_action(
                vector=state.vector, front=state.front,
                right=state.right, back=state.back,
                left=state.left, raycast=state.raycast)

        # action = np.random.choice(behavior_policy.shape[0], p=behavior_policy)
        action = np.argmax(behavior_policy)
        next_state, reward, done = env.step(convert_action(action))

        score += reward
        state = next_state

        if done:
            print(score)
            state = env.reset()
            episode += 1
            score = 0


if __name__ == '__main__':
    main()
