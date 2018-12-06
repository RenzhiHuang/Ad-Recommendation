import numpy as np
import random

def UCB(data, partial):
	'''
	data: 50 x 32657 numpy array
	partial: boolean to indicate if it is partial feedback or full feedback
	'''
	m,T = data.shape
	# the mean
	mu = data[:,0]
	# the test number 
	n = np.ones(m)
	regret = np.zeros(T)
	for t in range(1,T):
		UCB = mu + np.sqrt(2*np.log(t)/n)
		observed_arm = np.argmax(UCB)
		y_t = data[observed_arm,t]

		if(partial):
			n[observed_arm] = n[observed_arm]+1
			mu[observed_arm] = mu[observed_arm]+(y_t-mu[observed_arm])/n[observed_arm]
		else:
			n = n+1
			mu = mu+(data[:,t]-mu)/n

		if(y_t == 0):
			regret[t] = regret[t-1]
		else:
			regret[t] = regret[t-1]+1
	return regret


def Thompson_sampling(data):
	'''
	data: 50 x 32657 numpy array
	'''
	m,T = data.shape
	S = np.zeros(m)
	F = np.zeros(m)
	regret = np.zeros(T)
	for t in range(1,T):
		theta = np.random.beta(S+1,F+1)
		i_t = np.argmax(theta)
		r_t = data[i_t,t]
		if(r_t==1):
			S[i_t]=S[i_t]+1
			regret[t] = regret[t-1]
		else:
			F[i_t]=F[i_t]+1
			regret[t] = regret[t-1]+1
	return regret













