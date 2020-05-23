import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from probability import distr

def regret_plot(summary,config):
	numRounds = len(summary.numPlays)

	fig1_p, (ax1, ax2) = plt.subplots(nrows = 2, ncols = 1)
	plt.ylabel('Cumulative (weak) Regret')
	ax1.plot(range(numRounds), summary.weakRegret_p,label='pursuer weak regret')
	ax1.plot(range(numRounds), [i+1 for i in range(numRounds)],label='pursuer linear regret')
	ax1.plot(range(numRounds), summary.regretBound_p, label = 'pursuer expected upper bound')  

	ax1.legend()


	np_weights_evader = np.array(summary.dist_evader)
	transpose = np_weights_evader.T
	weights_evader = transpose.tolist()

	for w in weights_evader:
		ax2.plot(range(numRounds),w)

	plt.ylabel('Weight')
	ax2.set_title('evader weight vector')
	fig1_p.savefig(config['plot_dir'] + 'regret_weights.png')
    

	fig2, ax3 = plt.subplots()
	plt.ylabel('reward')
	ax3.plot(range(numRounds), [r/(i+1) for i,r in enumerate(summary.cumulativeRewardVec_p)], \
		label = 'pursuer average reward')
	ax3.plot(range(numRounds), [r/(i+1) for i,r in enumerate(summary.cumulativeRewardVec_e)], \
		label = 'evader average reward')

	average_plays = sum(summary.numPlays) / numRounds
	reward_eq_p = average_plays  / config['numActions']
	
	ax3.plot(range(numRounds), [reward_eq_p for i in range(numRounds)], \
		label = 'pursuer equilibrium reward')
	ax3.plot(range(numRounds), [1 - reward_eq_p for i in range(numRounds)], \
		label = 'evader equilibrium reward')

	ax3.set_title("average reward")
	ax3.legend()
	fig2.savefig(config['plot_dir'] + 'averageReward.png')