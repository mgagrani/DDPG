import torch
import numpy as np


class Rollout_storage(object):
    
    def __init__(self):
        
        self.reset()
    
    def reset(self):
        
        self.obs = []
        self.action = []
        self.next_obs = []
        self.reward = []
        self.done = []
    
    
    def append(self,obs,action,reward,next_obs,done):
        
        self.obs.append(obs.reshape(1,-1))
        self.action.append(action.reshape(1,-1))
        self.reward.append(reward.reshape(1,-1))
        self.next_obs.append(next_obs.reshape(1,-1))
        self.done.append(done.reshape(1,-1))
    
    
    def sample(self,batch_size):
        
        if len(self.obs)<batch_size:
            obs = torch.cat(self.obs[:],dim=0)
            action = torch.cat(self.action[:],dim=0)
            reward = torch.cat(self.reward[:],dim=0)
            next_obs = torch.cat(self.next_obs[:],dim=0)
            done = torch.cat(self.done[:],dim=0)
            
            return (obs,action,reward,next_obs,done)
        
        else:
            idxs = np.random.choice(len(self.obs),batch_size,replace=False)
            obs = torch.cat([self.obs[i] for i in idxs],dim=0)
            action = torch.cat([self.action[i] for i in idxs],dim=0)
            reward = torch.cat([self.reward[i] for i in idxs],dim=0)
            next_obs = torch.cat([self.next_obs[i] for i in idxs],dim=0)
            done = torch.cat([self.done[i] for i in idxs],dim=0)
            
            return (obs,action,reward,next_obs,done)
            
        