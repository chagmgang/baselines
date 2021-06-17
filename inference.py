from baselines.env.drone import Drone
from baselines.agent.drone_impala import DroneAgent

import tensorflow as tf
import numpy as np

import time

from baselines.misc import convert_action

def main():

    trajectory = 20

    actor = DroneAgent(
            trajectory=trajectory,
            model_name='learner',
            learner_name='learner')

    sess = tf.Session()
    actor.set_session(sess)
    actor.load_weights('saved/model-71000')

    episode = 0
    score = 0
    
    env = Drone(
            time_scale=0.1,
            port=11000,
            filename='/Users/chageumgang/Desktop/baselines/mac.app')

    state = env.reset()

    while True:

        _, behavior_policy, _ = actor.get_policy_and_action(
                vector=state.vector, front=state.front,
                right=state.right, back=state.back,
                left=state.left, raycast=state.raycast)

        action = np.argmax(behavior_policy)

        next_state, reward, done = env.step(convert_action(action))

        score += reward
        state = next_state

        if done:

            state = env.reset()
            episode += 1
            score = 0


if __name__ == '__main__':
    main()
