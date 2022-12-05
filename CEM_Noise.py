import numpy as np
from tetris import TetrisEnv
import matplotlib.pyplot as plt
import pdb

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
    while True:
        t+=1
        action_state_dict = env.get_next_states()
        next_val=[]
        for action in action_state_dict:
            feature = (action_state_dict.get(action))
            next_val.append(np.squeeze(weights.T.dot(feature)))

        index = np.argmax(next_val)
        next_action=best_state(action_state_dict,index)

        _, reward, done, _=env.step(next_action)
        reward_sum+=reward
        if done:
            #print("finished episode, got reward:{}".format(reward_sum))
            break
    return reward_sum

def get_constant_noise(step):
    return 0.5



"""
cem algorithm
iters : number of trainning iterations
weights_pop : population of weights
weights_dim : dimension of weights
sigma : standard deviation of additive noise
mean : mean of weights
elite_weights : weights of elite
mu :mean of weights after interpolation
elite_percent : percentage of elite weights
"""
weights_dim = 7
goal = 10000

start_num_players = 25
end_num_players = 20
start_num_episodes = 20
end_num_episodes = 25

init_scaling = 100
num_players = start_num_players
elite_percent = 0.2
num_elite = max(int(num_players * elite_percent), weights_dim)
num_episodes = start_num_episodes
regularize_iters = 50  # iteration to stop regularizing covariance matrix
interpolate_percent = 0.7
num_best = 4

iters = 501
env = TetrisEnv()
i = 0
mu = np.zeros(weights_dim)
cov = init_scaling * np.eye(weights_dim)
prev_mu = mu.copy()
prev_cov = cov.copy()
elite_weights = []  
prev_best = []

max_lines = []
mean_lines = []
var_lines = []
avg_trace_cov = []
for it in range(iters):
    prev_mu = mu.copy()
    prev_cov = cov.copy()
    weights = np.random.multivariate_normal(mu, cov, num_players)
    if len(prev_best) != 0:
        weights = np.concatenate([prev_best, weights])
        num_players += num_best
    lines_cleared = np.zeros(num_players)
    lines_cleared = [noisy_reward(env,w) for w in weights]
    print(lines_cleared)
    elite_rewards_and_weights = sorted(
        zip(lines_cleared, weights), key=lambda pair: pair[0]
    )[-num_elite:]
    elite_rewards = np.array([reward for reward, _ in elite_rewards_and_weights])
    elite_weights = np.array([weight for _, weight in elite_rewards_and_weights])
    prev_best = elite_weights[-num_best:]
    mu = np.mean(elite_weights, axis=0)
    cov = np.cov(elite_weights, rowvar=False)
    cov += max((regularize_iters - it) / 10, 0) * np.eye(weights_dim)
    mu = interpolate_percent * mu + (1.0 - interpolate_percent) * prev_mu
    cov = interpolate_percent * cov + (1.0 - interpolate_percent) * prev_cov
    elite_lines_cleared = np.mean(elite_rewards)

    percent_finished = min(elite_lines_cleared / goal, 1)
    num_players = start_num_players + int(
        percent_finished * (end_num_players - start_num_players)
    )
    num_episodes = start_num_episodes + int(
        percent_finished * (end_num_episodes - start_num_episodes)
    )
    lines_cleared = np.array(lines_cleared)
    max_lines.append(lines_cleared.max())
    mean_lines.append(lines_cleared.mean())
    var_lines.append(lines_cleared.var())
    avg_trace_cov.append(np.trace(cov) / weights_dim)
    print("iteration: {}, max lines: {}".format(it, max_lines[-1]))
    print("weights: {}".format(mu))
    # print("elite weights: {}".format(elite_weights))
    print("Mean Lines Cleared: ", lines_cleared.mean())

    
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(1, len(mean_lines)+1), mean_lines, label="Mean Lines Cleared")
plt.plot(np.arange(1, len(max_lines)+1), max_lines, label="Max Lines Cleared")
plt.legend(loc="upper left")
plt.xlabel('Iteration #')
plt.savefig('lines.png')    
plt.show()

