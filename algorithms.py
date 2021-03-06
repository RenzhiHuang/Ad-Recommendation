import numpy as np
import random

def epsilon_greedy(data, best_arm, epsilon=None):

	m,T = data.shape
	n = np.zeros(m)
	mu = np.zeros(m)

	regret = np.ones(T)
	regret_t = np.ones(T)
	reward = np.zeros(T)
	reward_best = np.zeros(T)
	for t in range(1,T):
		if epsilon is None:
			epsilon = 1/t
		if np.random.random()>epsilon:
			if max(mu)==0:
				choice = np.random.randint(m)
			else:
				choice = np.argmax(mu)
		else:
			choice = np.random.randint(m)

		reward[t] = reward[t-1]+data[choice,t]
		reward_best[t] = reward_best[t-1] + data[best_arm,t]
		regret[t] = (reward_best[t] - reward[t])/t
		regret_t[t] = regret[t]/t

		n[choice] += 1
		mu[choice] += ((n[choice]-1)*mu[choice]+data[choice,t])/n[choice]

	return regret, regret_t, reward

def UCB(data, partial,alpha,best_arm):
	'''
	data: 50 x 32657 numpy array
	partial: boolean to indicate if it is partial feedback or full feedback
	'''
	m,T = data.shape
	# the mean
	mu = data[:,0].astype(np.float)
	# the test number 
	n = np.ones(m)
	regret = np.zeros(T)
	regret_t = np.zeros(T)
	reward_UCB = np.zeros(T)
	reward_best = np.zeros(T)
	for t in range(1,T):
		UCB = mu + np.sqrt(alpha*np.log(t)/2/n)
		i_t = np.argmax(UCB)
		r_t = data[i_t,t]
		reward_UCB[t] = reward_UCB[t-1] + r_t
		reward_best[t] = reward_best[t-1] + data[best_arm,t]
		# regret[t] = regret[t-1]+(reward_best[t] - reward_UCB[t])/t
		regret[t] = (reward_best[t] - reward_UCB[t])/t
		regret_t[t] = regret[t]/t
		if(partial):
			n[i_t] = n[i_t]+1
			mu[i_t] = mu[i_t]+(r_t-mu[i_t])/n[i_t]
		else:
			n = n+1
			mu = mu+(data[:,t]-mu)/n

	return regret, regret_t, reward_UCB


def UCB_pro(data, alpha, best_arm, epsilon=None):
	'''
	data: 50 x 32657 numpy array
	partial: boolean to indicate if it is partial feedback or full feedback
	'''
	m,T = data.shape
	# the mean
	mu = data[:,0].astype(np.float)
	# the test number 
	n = np.ones(m)
	regret = np.zeros(T)
	regret_t = np.zeros(T)
	reward_UCB = np.zeros(T)
	reward_best = np.zeros(T)
	for t in range(1,T):
		UCB = mu + np.sqrt(alpha*np.log(t)/2/n)

		if epsilon is None:
			epsilon = 1/t

		if t <500:
			if np.random.random()>epsilon:
				if max(mu)==0:
					i_t = np.random.randint(m)
				else:
					i_t = np.argmax(UCB)
			else:
				i_t = np.random.randint(m)
		else:
			i_t = np.argmax(UCB)
		r_t = data[i_t,t]

		reward_UCB[t] = reward_UCB[t-1] + r_t
		reward_best[t] = reward_best[t-1] + data[best_arm,t]
		regret[t] = (reward_best[t] - reward_UCB[t])/t
		regret_t[t] = regret[t]/t

		n[i_t] = n[i_t]+1
		mu[i_t] = mu[i_t]+(r_t-mu[i_t])/n[i_t]


	return regret, regret_t, reward_UCB


def Thompson_sampling(data,partial,best_arm):
	'''
	data: 50 x 32657 numpy array
	partial: boolean to indicate if it is partial feedback or full feedback
	'''
	m,T = data.shape
	S = np.zeros(m)
	F = np.zeros(m)
	regret = np.zeros(T)
	regret_t = np.zeros(T)
	reward_Tho = np.zeros(T)
	reward_best = np.zeros(T)
	for t in range(0,T):
		theta = np.random.beta(S+1,F+1)
		i_t = np.argmax(theta)
		r_t = data[i_t,t]

		if(t>0):
			reward_Tho[t] = reward_Tho[t-1] + r_t
			reward_best[t] = reward_best[t-1] + data[best_arm,t]
			regret[t] = (reward_best[t] - reward_Tho[t])/t
			regret_t[t] = regret[t]/t
		else:
			reward_Tho[t] = r_t
			reward_best[t] = data[best_arm,t]
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
	return regret, regret_t, reward_Tho













