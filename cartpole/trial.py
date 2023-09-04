import gym
import popgym
import numpy as np
# env_classes = popgym.ALL_ENVS.keys()
# print(env_classes)
from popgym.envs.stateless_cartpole import StatelessCartPole
from qlearning import DQNSolver

if __name__ == "__main__":
    env = StatelessCartPole()
    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.n
    dqn_solver = DQNSolver(observation_space, action_space)
    run = 0
    done = False
    obs, info = env.reset(return_info=True)
    reward = -float("inf")
    # game.render()
    while True:
        run += 1
        obs = env.reset()
        obs = np.reshape(obs, [1, observation_space])
        done = False

        step = 0
        # print(run)
        # env.render()

        while not done:
            step += 1
            # env.render()

            action = dqn_solver.act(obs)
            obs_next, reward, done, info = env.step(int(action))
            reward = reward if not done else -reward
            obs_next = np.reshape(obs_next, [1, observation_space])
            dqn_solver.remember(obs, action, reward, obs_next, done)
            obs = obs_next

            # env.render()
            if done:
                print("reward:", step)
                break
            dqn_solver.experience_replay()

        # print(obs)
