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
path = '/Users/galaxydirector/Desktop/Ad-Recommendation/yahoo_ad_clicks.csv'
# path = './yahoo_ad_clicks.csv'
partial = True # True to be partial feedback, False to be full feedback
alpha = 2 # larger alpha indicates more preference to exploration

# data import
data = data_import(path)
print(type(data)) # numpy array
print(data.shape) # 50 x 32657

# gain some knowledge about the data
m,T = data.shape
mu = np.mean(data,axis = 1)
best_choice = np.argmax(mu)
highest_mean = np.max(mu)
#print(highest_mean)
#print(np.sort(mu))

round_sum = np.sum(data,axis=0)
print(np.max(round_sum))


# test algorithms
# regret, regret_t,reward = epsilon_greedy(data, epsilon=None)
# regret, regret_t,reward = UCB(data,partial,alpha)
regret, regret_t, reward = Thompson_sampling(data,partial)

# for e in range(2,10,2):
# 	regret, regret_t,reward = epsilon_greedy(data, epsilon=e/10)
# 	output_path = './greedy_{}'.format(e)
output_path = './Thompson'
plot_(reward, output_path, num_skip = 20, name = 'reward')
plot_(regret, output_path, num_skip = 20, name = 'regret')
plot_(regret_t, output_path, num_skip = 20, name = 'mean_regret')

print(reward[-10:])
print(regret[-10:])
print(regret_t[-10:])
# plt.plot(regret_t[20:])
# plt.show()


