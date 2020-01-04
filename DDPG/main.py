from utils import *
import torch
import torch.nn as nn
import gym
import numpy as np
import argparse
from DDPG import DDPG_agent
from nets import actor_net,critic_net
from buffer import Rollout_storage
#import matplotlib.pyplot as plt


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name',default='InvertedPendulum-v2')
    parser.add_argument('--seed',default=0,type=int)
    parser.add_argument('--discount',default=0.99)
    parser.add_argument('--actor_lr',default=1e-3)
    parser.add_argument('--critic_lr',default=1e-3)
    parser.add_argument('--tau',default=0.05)
    parser.add_argument('--noise_var',default=0.1)
    parser.add_argument('--batch_size',default=100,type=int)
    parser.add_argument('--max_steps',default=100000,type=int)
    parser.add_argument('--eval_steps',default=5000,type=int)
    parser.add_argument('--log_steps',default=2000,type=int)
    parser.add_argument('--eval_episodes',default=10,type=int)
    parser.add_argument('--update_freq',default=1,type=int)
    parser.add_argument('--t_explore',default=1000,type=int)

    args = parser.parse_args()

    env = gym.make(args.env_name)
    #eval_env = gym.make(env_name)

    high_action = env.action_space.high
    low_action = env.action_space.low

    action_dim = high_action.shape[0]
    obs = env.reset()
    obs_dim = obs.shape[0]

    actor = actor_net(obs_dim=obs_dim,action_dim=action_dim,h=30)
    critic = critic_net(obs_dim,action_dim,h=30)

    target_actor = actor_net(obs_dim=obs_dim,action_dim=action_dim,h=30)
    target_critic = critic_net(obs_dim,action_dim,h=30)

    target_actor.load_state_dict(actor.state_dict())
    target_critic.load_state_dict(critic.state_dict())

    rollout_storage = Rollout_storage()
    action_range = (low_action[0],high_action[0])

    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'

    actor = actor.to(device)
    critic = critic.to(device)
    target_actor = target_actor.to(device)
    target_critic = target_critic.to(device)

    agent = DDPG_agent(actor,critic,target_actor,target_critic,rollout_storage,args.discount,
                args.tau,args.update_freq,args.noise_var,action_range,
                args.actor_lr,args.critic_lr,args.batch_size,device)

    agent,returns = train(agent,args.env_name,args.max_steps,args.eval_episodes,
                         args.eval_steps,args.log_steps,args.seed,args.t_explore)

    
    
    log_policy_rollout(agent,args.env_name)
    #plt.plot(returns)
    eval_return = eval_agent(agent,args.env_name,args.seed,args.eval_episodes)
    print('Evaluation reward at the end of training is {}'.format(eval_return))
    print('The mean of the returns is {}'.format(np.mean(returns)))
    print('-'*7 + 'Done' + '-'*7)


    




def train(agent,env_name,max_steps,num_eval_episodes,eval_steps,log_steps,seed,t_explore):
    
    
    epoch_episode_rewards = []
    episode_reward = 0.0
    num_train_episode = 0
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    env = gym.make(env_name)
    env.seed(seed)
    obs = env.reset()
    
    all_eval_returns = []
    
    for t in range(max_steps):
        
        if t<t_explore:
            action = env.action_space.sample()
        else:
            action = agent.act(obs,apply_noise=True)
        
        new_obs,reward,done,_ = env.step(action)
        agent.store_transition(obs,action,reward,new_obs,done)
        obs = new_obs
                
        episode_reward += reward
                
        if done:
            epoch_episode_rewards.append(episode_reward)
            episode_reward = 0.0
            num_train_episode += 1
            obs = env.reset()
            done = False
                    
                
        ## Training the actor and critic network
        if t>t_explore:    
            critic_loss,actor_loss = agent.train()
            #epoch_critic_loss.append(critic_loss)
            #epoch_actor_loss.append(actor_loss)
                
            
            
        if t%eval_steps == 0 and t>=t_explore:
            print('-'*5 + 'Evaluating' + '-'*5)
            
            ## Evaluating the current performance
            
            eval_return = eval_agent(agent,env_name,seed,num_eval_episodes)
            all_eval_returns.append(eval_return)
            
            print('Mean reward at time {}/{} is {}'.format(t,max_steps,eval_return))
    
        if t%log_steps == 0 and t>0:
            print('Mean training reward at time {} is {}'.format(t,np.mean(epoch_episode_rewards)))
            print('Number of completed episodes {}'.format(num_train_episode))
            epoch_episode_rewards = []
            num_train_episode = 0
    
    
    return agent,all_eval_returns






if __name__ == '__main__':
    main()


