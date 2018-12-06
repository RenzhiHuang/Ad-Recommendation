import numpy as np

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



