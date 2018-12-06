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


# test algorithms
#regret = UCB(data,partial)
regret = Thompson_sampling(data,partial)
print(regret)
plt.plot(regret)
plt.show()