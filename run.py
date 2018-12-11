from data_import import *
from algorithms import *
import os
import matplotlib.pyplot as plt

def plot_(data, output_path, num_skip = 20, name = 'regret'):
	"""
	"""
	if not os.path.exists(output_path):
		os.makedirs(output_path)

	plt.plot(data[num_skip:])
	plt.xlabel('iteration')
	plt.ylabel(name)
	plt.title(name)
	path = os.path.join(output_path, '{}.png'.format(name))
	plt.savefig(path)
	plt.close()

# parameters to be set
#path = '/Users/galaxydirector/Desktop/Ad-Recommendation/yahoo_ad_clicks.csv'
path = './yahoo_ad_clicks.csv'
partial = True # True to be partial feedback, False to be full feedback
# partial = False
alpha = 2 # larger alpha indicates more preference to exploration

# data import
data = data_import(path)
print(type(data)) # numpy array
print(data.shape) # 50 x 32657

# gain some knowledge about the data
m,T = data.shape
reward =  np.sum(data,axis = 1)
mu = np.mean(data,axis = 1)
best_arm = np.argmax(mu)
mu_star = np.max(mu)
print("The best arm has the reward of %d")%np.max(reward)
print(best_arm)

round_sum = np.sum(data,axis=0)
print(np.max(round_sum))


# test algorithms
# regret, regret_t,reward = epsilon_greedy(data, best_arm, epsilon=None)
regret, regret_t,reward = UCB(data,partial,alpha,best_arm)
# regret, regret_t, reward = Thompson_sampling(data,partial, best_arm)
# regret, regret_t,reward = UCB_pro(data, alpha, best_arm, epsilon=0.8)
output_path = './UCB'

# for e in range(2,10,2):
# 	# regret, regret_t,reward = epsilon_greedy(data, best_arm, epsilon=e/10)
# 	regret, regret_t,reward = UCB_pro(data, alpha, best_arm, epsilon=e/10)
# 	output_path = './ucbpro_{}'.format(e)

plot_(reward, output_path, num_skip = 20, name = 'reward')
plot_(regret, output_path, num_skip = 20, name = 'regret')
plot_(regret_t, output_path, num_skip = 20, name = 'mean_regret')

print("terminal reward",regret[-1])

#print(reward[-10:])
#print(regret[-10:])
#print(regret_t[-10:])
# plt.plot(regret_t[20:])
# plt.show()


