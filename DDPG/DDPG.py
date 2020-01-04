
import torch


class DDPG_agent(object):
    
    def __init__(self,actor,critic,target_actor,target_critic,rollout_storage,gamma,tau,update_freq,explore_noise_std,
                 action_range,actor_lr,critic_lr,batch_size,device):
        
        self.actor = actor
        self.critic = critic
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.gamma = gamma
        self.batch_size = batch_size
        self.tau = tau
        self.train_steps = 0
        self.update_freq = update_freq
        self.explore_noise_std = explore_noise_std
        self.action_range = action_range
        
        self.rollout_storage = rollout_storage
        
        self.actor_optim = torch.optim.Adam(self.actor.parameters(),lr=self.actor_lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(),lr=self.critic_lr)
        
        self.target_critic = target_critic
        self.target_actor = target_actor

        self.device = device
        
        
        
    def act(self,obs,apply_noise):
        
        if not isinstance(obs,torch.Tensor):
            obs = torch.tensor(obs,dtype=torch.float32).to(self.device)
        
        action = self.actor(obs)
        
        if apply_noise:
            action = torch.clamp(action+self.explore_noise_std*torch.randn_like(action),min=self.action_range[0],max=self.action_range[1])
        else:
            action = torch.clamp(action,min=self.action_range[0],max=self.action_range[1])
            
        return action.data.cpu().numpy()
    
    
    def store_transition(self,obs,action,reward,new_obs,done):
        
        obs = torch.tensor(obs,dtype=torch.float32).to(self.device)
        action = torch.tensor(action,dtype=torch.float32).to(self.device)
        reward = torch.tensor(reward,dtype=torch.float32).to(self.device)
        new_obs = torch.tensor(new_obs,dtype=torch.float32).to(self.device)
        done = torch.tensor(done,dtype=torch.bool).to(self.device)
        
        self.rollout_storage.append(obs,action,reward,new_obs,done)
    
    
    def train(self):
        
        data = self.rollout_storage.sample(self.batch_size)
        
        obs,action,reward,next_obs,terminal = data
        
        with torch.no_grad():
            target_q = reward + self.gamma*(~terminal)*(self.target_critic(next_obs,self.target_actor(next_obs)))
           
        
        # Updating the critic and actor networks
        
        critic_loss = torch.mean((target_q-self.critic(obs,action))**2)
        
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()
        
        actor_loss = -torch.mean(self.critic(obs,self.actor(obs)))
        
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()
        
        
        # Updating the target networks
        
        if self.train_steps%self.update_freq==0 and self.train_steps != 0 :
            
            for target_param,param in zip(self.target_critic.parameters(),self.critic.parameters()):
                target_param.data.copy_(self.tau*param.data + (1-self.tau)*target_param.data)
            
            for target_param,param in zip(self.target_actor.parameters(),self.actor.parameters()):
                target_param.data.copy_(self.tau*param.data + (1-self.tau)*target_param.data)
        
        self.train_steps += 1
        
        return critic_loss,actor_loss
    
        
            