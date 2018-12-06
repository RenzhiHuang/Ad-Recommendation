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

# test algorithms
regret = UCB(data,partial)
print(regret)
plt.plot(regret)
plt.show()