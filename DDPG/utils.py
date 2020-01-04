import glob
import io
import base64

from IPython.display import HTML
from IPython import display as ipythondisplay

import gym
from gym.wrappers import Monitor



def wrap_env(env):
    env = Monitor(env, './video', force=True)
    return env



def show_video():
    mp4list = glob.glob('video/*.mp4')
    if len(mp4list) > 0:
        mp4 = mp4list[0]
        video = io.open(mp4, 'r+b').read()
        encoded = base64.b64encode(video)
        ipythondisplay.display(HTML(data='''<video alt="test" autoplay 
                loop controls style="height: 400px;">
                <source src="data:video/mp4;base64,{0}" type="video/mp4" />
             </video>'''.format(encoded.decode('ascii'))))
    else:
        print("Could not find video")




def log_policy_rollout(agent,env_name):
    
    env = wrap_env(gym.make(env_name))
    
    observation = env.reset()

    done = False
    episode_reward = 0
    episode_length = 0

    # Run until done == True
    while not done:
      # Take a step
        
        action = agent.act(observation,apply_noise=False)
        observation, reward, done, _ = env.step(action)

        episode_reward += reward
        episode_length += 1

    print('Total reward:', episode_reward)
    print('Total length:', episode_length)

    env.close()
    
    show_video()


def eval_agent(agent,env_name,seed,num_episodes):
    
    eval_env = gym.make(env_name)
    eval_env.seed(seed)
    
    
    
    total_reward = 0.0
    
    for _ in range(num_episodes):
        
        obs = eval_env.reset()
        done = False
        
        while not done:
            action = agent.act(obs,apply_noise=False)
            new_obs,reward,done,_ = eval_env.step(action)
            total_reward += reward
            obs = new_obs
            
            
    
    return total_reward/num_episodes
