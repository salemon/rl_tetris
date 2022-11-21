import numpy as np
from tetris import TetrisEnv
import matplotlib.pyplot as plt
env = TetrisEnv()

def best_state(action_state_dict, index):
        max_score = None
        best_action = None
        best_state = None
       
        next_actions, next_states = zip(*action_state_dict.items())
        best_action = next_actions[index]
        return best_action

def noisy_reward(env,weights):
    reward_sum = 0
    state = env.reset()
    t = 0 
    cleared = 0
    while True:
        t+=1
        action_state_dict = env.get_next_states()
        next_val=[]
        for action in action_state_dict:
            feature = (action_state_dict.get(action))
            next_val.append(np.squeeze(weights.T.dot(feature)))

        index = np.argmax(next_val)
        next_action=best_state(action_state_dict,index)

        _, reward, done, cleared_current_turn=env.step(next_action)
        if done:
            print("finished episode")
            break
        else:
            cleared = cleared + cleared_current_turn
            reward_sum=reward
            # print("Cleared Lines: ", cleared)
    
    env.render()
    return reward_sum

def get_constant_noise(step):
    return 0.3

n_iter = 20
i = 0
n_weights = 25
weights_dim = 7

std = np.array([2]*weights_dim)
mean = np.array([-13.64655776,   7.03165802,  -6.67430539, -10.22211962, -23.80880996,
  -7.71383025,  -7.21648756])

weights_pop = np.array([mean]*20) #modify this to change the number of testing times
#print(weights_pop)
env = TetrisEnv()

reward = [noisy_reward(env,weights) for weights in weights_pop]
print('lines cleared during 20 games:',reward)
avg_reward = np.mean(reward)
print('average lines cleared:',avg_reward)



    
