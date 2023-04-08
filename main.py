from maddpg.MAAC import MAAC
from maddpg.params import scale_reward
import numpy as np
import torch as th
import wandb
# import pressureplate
from gym.envs.registration import register
import gym
import argparse
parser = argparse.ArgumentParser(description=None)
parser.add_argument('-e', '--env', default='soccer', type=str)

args = parser.parse_args()

register(
            id='multigrid-soccer-v0',
            entry_point='gym_multigrid.envs:SoccerGame4HEnv10x15N2',
        )


register(
            id='multigrid-collect-v0',
            entry_point='gym_multigrid.envs:CollectGame4HEnv10x10N2',
        )


# do not render the scene
e_render = False

food_reward = 10.
poison_reward = -1.
encounter_reward = 0.01
n_coop = 2


world = gym.make('multigrid-soccer-v0')

# vis = visdom.Visdom(port=5274)
reward_record = []

np.random.seed(1234)
th.manual_seed(1234)
world.reset()
n_agents = len(world.agents)
n_states = np.prod(world.observation_space.shape) #TODO 이것도 수정했습니다.
n_actions = world.action_space.n
capacity = 1000000
batch_size = 1000

n_episode = int(3e6)
max_steps = 1000
episodes_before_train = 100

win = None
param = None

maddpg = MAAC(n_agents, n_states, n_actions, batch_size, capacity,
                episodes_before_train)
wandb.init(project="baebae_0409")
wandb.run.name = f"baebaerun_maac"

FloatTensor = th.cuda.FloatTensor if maddpg.use_cuda else th.FloatTensor
for i_episode in range(n_episode):
    obs = world.reset()
    obs = np.stack(obs)
    if isinstance(obs, np.ndarray):
        obs = th.from_numpy(obs).float()
    total_reward = 0.0
    rr = np.zeros((n_agents,))
    for t in range(max_steps):
        log = {}
        # render every 100 episodes to speed up training
        if i_episode % 20 == 0 and e_render:
            world.render()
        obs = obs.type(FloatTensor)
        #TODO: action space 어떻게 구성된거임
        # action = maddpg.select_action(obs).data.cpu()
        # actions = {agent: maddpg.select_action(obs[agent]).data.cpu().numpy() for agent in
        #            world.agents}  #
        agents_actions = maddpg.select_action(obs.reshape((n_agents,-1))).data.cpu().numpy()
        # agents_actions = maddpg.select_action(obs).data.cpu().numpy()
        actions = np.argmax(agents_actions, axis=1)

        obs_, reward, done, _ = world.step(actions)
        # obs_, reward, done, _ = world.step((action*0.01).numpy())

        reward = np.stack(reward)
        reward = th.from_numpy(reward).float()
        if t != max_steps - 1:
            next_obs = obs_
            next_obs = np.stack(next_obs)
            if isinstance(next_obs, np.ndarray):
                next_obs = th.from_numpy(next_obs).float()
        else:
            next_obs = None

        total_reward += reward.sum()
        rr += reward.cpu().numpy()
        
        #make actions to be one-hot vectors
        actions = np.eye(n_actions)[actions].reshape((-1,))
        
        maddpg.memory.push(obs.data, th.from_numpy(np.stack(actions)).float(), next_obs, reward)
        obs = next_obs

        c_loss, a_loss = maddpg.update_policy()

        # add reward, c_loss, a_loss to log
        log['t/reward'] = reward.sum()
        log['t/c_loss'] = c_loss
        log['t/a_loss'] = a_loss
        wandb.log(log)

        if done:
            # print('done: {} {} {} {} {}'.format(*done))
            # print('truncated: {} {} {} {} {}'.format(*truncated))
            break
    maddpg.episode_done += 1
    print('Episode: %d, reward = %f' % (i_episode, total_reward))
    reward_record.append(total_reward)

    if maddpg.episode_done == maddpg.episodes_before_train:
        print('training now begins...')
        print('MADDPG on WaterWorld\n' +
              'scale_reward=%f\n' % scale_reward +
              'agent=%d' % n_agents +
              ', coop=%d' % n_coop +
              ' \nlr=0.001, 0.0001, sensor_range=0.3\n' +
              'food=%f, poison=%f, encounter=%f' % (
                  food_reward,
                  poison_reward,
                  encounter_reward))
wandb.finish()

world.close()