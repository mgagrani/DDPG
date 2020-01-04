import torch.nn as nn
import torch


class actor_net(nn.Module):
    
    def __init__(self,obs_dim,action_dim,h=256):
        super(actor_net,self).__init__()
        self.actor = nn.Sequential(nn.Linear(obs_dim,h),nn.ReLU(),nn.Linear(h,h),nn.ReLU(),nn.Linear(h,action_dim),nn.Tanh())
        
    def forward(self,state):
        return self.actor(state)
    


class critic_net(nn.Module):
    
    def __init__(self,obs_dim,action_dim,h=256):
        super(critic_net,self).__init__()
        self.critic = nn.Sequential(nn.Linear(obs_dim+action_dim,h),nn.ReLU(),nn.Linear(h,h),nn.ReLU(),nn.Linear(h,1))
        #self.fc1 = nn.Linear(state_dim+action_dim,h)
        #self.fc2 = nn.Linear(h,h)
        #self.fc3 = nn.Linear(h,1)
        
    def forward(self,state,action):
        
        x1 = torch.cat([state,action],dim=1)
        #x2 = F.relu(self.fc1(x1))
        #x3 = F.relu(self.fc2(x2))
        #x4 = self.fc3(x3)
        
        return self.critic(x1)
