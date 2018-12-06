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
	regret = np.ones(T)
	regret_t = np.ones(T)
	reward = np.zeros(T)
	for t in range(1,T):
		UCB = mu + np.sqrt(2*np.log(t)/n)
		i_t = np.argmax(UCB)
		r_t = data[i_t,t]
		reward[t] = reward[t-1]+r_t
		regret[t] = regret[t-1]+ (np.max(mu)-mu[i_t])
		regret_t[t] = regret[t]/t
		if(partial):
			n[i_t] = n[i_t]+1
			mu[i_t] = mu[i_t]+(r_t-mu[i_t])/n[i_t]
		else:
			n = n+1
			mu = mu+(data[:,t]-mu)/n

	return regret, regret_t, reward


def Thompson_sampling(data,partial):
	'''
	data: 50 x 32657 numpy array
	partial: boolean to indicate if it is partial feedback or full feedback
	'''
	m,T = data.shape
	S = np.zeros(m)
	F = np.zeros(m)
	regret = np.ones(T)
	regret_t = np.ones(T)
	reward = np.zeros(T)
	for t in range(0,T):
		theta = np.random.beta(S+1,F+1)
		i_t = np.argmax(theta)
		r_t = data[i_t,t]
		if(t>0):
			reward[t] = reward[t-1]+r_t
			mu = S/(S+F)
			regret[t] = regret[t-1]+ (np.max(mu)-mu[i_t])
			regret_t[t] = regret[t]/t
		else:
			reward[t] = r_t
		if(r_t==1):
			if(partial):
				S[i_t]=S[i_t]+1
			else:
				S = S + data[:,t]
		else:
			if(partial):
				F[i_t]=F[i_t]+1
			else:
				F = F + (1-data[:,t])
	return regret, regret_t, reward













