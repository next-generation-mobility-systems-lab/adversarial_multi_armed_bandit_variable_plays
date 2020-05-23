import math
import random
import numpy as np
import yaml
import os
import matplotlib as mpl
import matplotlib.pyplot as plt

from probability import distr_multiPlays, distr, draw
from utility_plot import regret_plot 
from utilities import getAlpha, find_indices, reward, updateState_p,updateState_e, DepRound, randomInt

class summary:
	choice_p = []
	choice_e = []
	rewardVector_p = []
	rewardVector_e = []
	numPlays = []
	
	cumulativeReward_p = 0
	cumulativeRewardVec_p = []
	cumulativeReward_e = 0
	cumulativeRewardVec_e = []

	weights_pursuer = []
	weights_evader = []
	dist_evader = []

	bestActionCumulativeReward_p = 0
	bestActionCumulativeReward_e = 0

	weakRegret_p = []
	weakRegret_e = []

	regretBound_p = []
	regretBound_e = []
	
	bestAction_p = []
	bestAction_e = []

class State_p:
	gamma_p = []
	S_0 = []

class State_e:
	gamma_e = []


def advGame(config,numPlays):
	numActions = config['numActions']
	numRounds = config['numRounds']

	rewardMax = config['rewardMax']
	rewardMin = config['rewardMin']

	State_p.weights_pursuer = [1.0] * numActions # initialize weight vector for the pursuer
	State_e.weights_evader = [1.0] * numActions # initialize weight vector for the evader

	t = 0

	if config['custom_gamma_pursuer']:
		State_p.gamma_p = config['gamma_p']

	else:
		State_p.gamma_p =  min([1, math.sqrt(numActions * numActions * numPlays_LB *math.log(numActions/numPlays_UB)\
			 / (( (math.e - 2) * numPlays_UB + numPlays_LB)* numPlays_UB * numRounds ) ) ])

	if config['custom_gamma_evader']:
		State_e.gamma_e =  min([1, math.sqrt(numActions * math.log(numActions) / ((math.e - 1) *  numRounds ) ) ])

	else:
		State_e.gamma_e = config['gamma_e']

	# entering the main loop
	while True:
		theSum_pursuer = sum(State_p.weights_pursuer)
		theSum_evader = sum(State_e.weights_evader)
		State_p.weights_pursuer = [w / theSum_pursuer for w in State_p.weights_pursuer] # normalize the weight vector of the pursuer
		State_e.weights_evader = [w / theSum_evader for w in State_e.weights_evader] # normalize the weight vector of the pursuer

		temp = (1.0 / numPlays - State_p.gamma_p / numActions) * float( 1.0 / (1.0 - State_p.gamma_p) )
		w_temp_p = State_p.weights_pursuer
		w_sorted_p = sorted(State_p.weights_pursuer, reverse=True)


		if w_sorted_p[0] >= temp * theSum_pursuer:
			State_p.alpha_t = getAlpha(temp, w_sorted_p)

			State_p.S_0 = find_indices(w_temp_p, lambda e: e >= State_p.alpha_t)

			for s in State_p.S_0:
				w_temp_p[s] = State_p.alpha_t


		else:
			State_p.S_0 = []


		State_p.probDist_p = distr_multiPlays(w_temp_p, numPlays, gamma = State_p.gamma_p)
		State_e.probDist_e = distr(State_e.weights_evader, gamma = State_e.gamma_e)

		assert False in np.isnan(np.array(State_p.probDist_p)), "Error, probability of pursuer must be a real number"
		assert False in np.isnan(np.array(State_e.probDist_e)), "Error, probability of evader must be a real number"

		State_p.choice_p = sorted(DepRound(State_p.probDist_p, k = numPlays))
		State_e.choice_e = draw(State_e.probDist_e)

		State_p.reward_p, State_e.reward_e, _ = reward(State_p.choice_p, State_e.choice_e)

		State_p.rewardFull_p =  [0.0] * numActions # initialize reward vector for the pursuer
		State_e.rewardFull_e = [0.0] * numActions # initialize reward vector for the evader

		for i in State_p.choice_p:
			for r in State_p.reward_p:
				State_p.rewardFull_p[i] = r

		State_e.rewardFull_e[State_e.choice_e] = State_e.reward_e


		State_p.scaledReward_p = [(r - rewardMin) / (rewardMax - rewardMin) for r in State_p.rewardFull_p] # reward vector scaled to 0,1, optional
		State_e.scaledReward_e = [(r - rewardMin) / (rewardMax - rewardMin) for r in State_e.rewardFull_e] # reward vector scaled to 0,1, optional


		updateState_p(State_p)
		updateState_e(State_e)

		State_p.cumulativeReward_p = sum(State_p.reward_p)
		State_e.cumulativeReward_e = State_e.reward_e
		assert State_p.cumulativeReward_p + State_e.cumulativeReward_e == 1, "Error, should be constant sum game!"
        
		yield State_p, State_e
		t = t + 1

		







if __name__ == '__main__':
	with open('config_search.yaml','r+') as f_search:
		config_search = yaml.load(f_search)

	for nA_idx, nA_val in enumerate(config_search['numActionsVec']):
		config_search['numActions'] = nA_val

        # iterates through all combinations


		for nR_idx, nR_val in enumerate(config_search['numRoundsVec']):
			for nPL_idx, nPL_val in enumerate(config_search['numPlays_LBVec']):
				for nPU_idx, nPU_val in enumerate(config_search['numPlays_UBVec']):

					config_search['numRounds'] = nR_val
					config_search['numPlays_LB'] = nPL_val
					config_search['numPlays_UB'] = nPU_val

        			# create the directories for recording results and summaries
					subscript = str(nA_idx) + str(nR_idx) + str(nPL_idx) + str(nPU_idx)
					config_search['summary_dir'] = config_search['summaries_dir'] + 'summary_' + subscript + '/'
					config_search['plot_dir'] = config_search['plots_dir'] + 'plot_' + subscript + '/'

					if not os.path.exists(config_search['summary_dir']):
						os.makedirs(config_search['summary_dir'])

					if not os.path.exists(config_search['plot_dir']):
						os.makedirs(config_search['plot_dir'])


					with open(config_search['plot_dir'] + 'config.yaml', 'w') as f:
						yaml.dump(config_search,f)                 

					t = 0

					numPlays_LB = nPL_val
					numPlays_UB = nPU_val
					numPlays_std = config_search['numPlays_std']
					# summary.numPlays = [random.randint(numPlays_LB, numPlays_UB) for _ in range(nR_val)] 		# list of number of plays that is uniformly distributed for the pursuer
					summary.numPlays = randomInt(numPlays_LB, numPlays_UB, numPlays_std, nR_val)

					try:
						for (State_p, State_e) in advGame(config_search, summary.numPlays[t]):
							summary.cumulativeReward_p += State_p.cumulativeReward_p
							summary.cumulativeRewardVec_p.extend([summary.cumulativeReward_p])
							summary.cumulativeReward_e += State_e.cumulativeReward_e
							summary.cumulativeRewardVec_e.extend([summary.cumulativeReward_e])

							summary.weights_pursuer.append(State_p.weights_pursuer)
							summary.weights_evader.append(State_e.weights_evader)
							summary.dist_evader.append(distr(State_e.weights_evader))

							summary.choice_p.append(State_p.choice_p)
							summary.choice_e.append([State_e.choice_e])
							summary.rewardVector_p.append(State_p.rewardFull_p)
							summary.rewardVector_e.append(State_e.rewardFull_e)

                            
							print("pursuer weights:(%s)\tevader weights(%s)\t" % \
								(', '.join(["%.3f" % r for r in distr_multiPlays(State_p.weights_pursuer, summary.numPlays[t])]),','.join(["%.3f" % r for r in distr(State_e.weights_evader)])))

							t += 1
							if t>= nR_val:
								break

						bestActionSet_p = sorted(range(config_search['numActions']), key=lambda action: sum([summary.rewardVector_p[t][action] \
							for t in range(nR_val)]), reverse=True)[:config_search['numPlays_UB']]
						bestActionSet_e = max(range(config_search['numActions']), key=lambda action: sum([summary.rewardVector_e[t][action] \
							for t in range(nR_val)]))

						log_p = open(config_search['summary_dir'] + 'log_pursuer.txt','w+')
						log_e = open(config_search['summary_dir'] + 'log_evader.txt','w+')

						for s in range(nR_val):
							summary.bestAction_p.append(bestActionSet_p[:summary.numPlays[s]])
							summary.bestActionCumulativeReward_p += sum([summary.rewardVector_p[s][i] for i in bestActionSet_p[:summary.numPlays[s]]])
							summary.bestActionCumulativeReward_e += summary.rewardVector_e[s][bestActionSet_e]
							
							summary.regretBound_p.append((1 + (math.e - 2)* config_search['numPlays_UB'] / config_search['numPlays_LB'])* State_p.gamma_p \
								* summary.bestActionCumulativeReward_p + (config_search['numActions'] * math.log(config_search['numActions']/config_search['numPlays_UB'])) / State_p.gamma_p)
							
							summary.regretBound_e.append((math.e - 1) * State_e.gamma_e * summary.bestActionCumulativeReward_e \
								+ (config_search['numActions'] * math.log(config_search['numActions'])) / State_e.gamma_e) 

							summary.weakRegret_p.extend([summary.bestActionCumulativeReward_p - summary.cumulativeRewardVec_p[s]])
							summary.weakRegret_e.extend([summary.bestActionCumulativeReward_e - summary.cumulativeRewardVec_e[s]])


							log_p.write("t:%d\tnumPlays:%d\tregret:%d\treward:%d\tweights:(%s)\r\n" % (s+1, summary.numPlays[s],summary.weakRegret_p[s],summary.cumulativeRewardVec_p[s],\
								', '.join(["%.3f" % weight for weight in distr_multiPlays(summary.weights_pursuer[s], summary.numPlays[s])])))

							log_e.write("t:%d\tregret:%d\treward:%d\tweights:(%s)\r\n" % (s+1, summary.weakRegret_e[s], summary.cumulativeRewardVec_e[s],\
								', '.join(["%.3f" % weight for weight in distr(summary.weights_evader[s])])))

						regret_plot(summary, config_search) # ploting

					except KeyboardInterrupt:
						pass


