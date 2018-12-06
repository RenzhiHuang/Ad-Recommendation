from data_import import *
from algorithms import *
import matplotlib.pyplot as plt

# parameters to be set
path = './yahoo_ad_clicks.csv'
partial = False # True to be partial feedback, False to be full feedback

# data import
data = data_import(path)
print(type(data)) # numpy array
print(data.shape) # 50 x 32657

# gain some knowledge about the data
m,T = data.shape
mu = np.mean(data,axis = 1)
best_choice = np.argmax(mu)
highest_mean = np.max(mu)
print(highest_mean)
print(np.sort(mu))

round_sum = np.sum(data,axis=0)
print(np.max(round_sum))


# test algorithms
regret, regret_t = UCB(data,partial)
#regret, regret_t = Thompson_sampling(data,partial)
print(regret)
plt.plot(regret_t)
plt.show()