from probability import distr_multiPlays, distr, draw
from DepRound import DepRound, DepRound1
import math
import random
from scipy.optimize import fsolve
import numpy as np
import yaml
import os

class summary:
	choice_p = []
	choice_e = []
	rewardVector_p = []
	rewardVector_e = []
	numPlays = []

class State_p:
	gamma_p = []
	S_0 = []

class State_e:
	gamma_e = []



def getAlpha(temp, w_sorted):
	# getAlpha calculates the alpha value for the sorted weight vector.
	sum_weight = sum(w_sorted)

	for i in range(len(w_sorted)):
		alpha = (temp * sum_weight) / (1 - i * temp)
		curr = w_sorted[i]

		if alpha > curr:
			alpha_exp = alpha
			return alpha_exp

		sum_weight = [s - curr for s in sum_weight]

	raise Exception('alpha not found')


def find_indices(lst, condition):
	# Function that returns the indices satisfying the condition function
	return [i for i, elem in enumerate(lst) if condition(elem)]

def reward(choice_p, choice_e):
	common_choice = list(set(choice_p).intersection([choice_e]))
	reward_e = 0.0
	reward_p = [0.0] * len(choice_p)

	if not common_choice:
		reward_e = 1.0 
	else:
		for i in common_choice:
			indx = choice_p.index(i)
			reward_p[indx] = 1.0

	return reward_p, reward_e, common_choice


def updateState_p(State_p):
	numActions = len(State_p.scaledReward_p)
	numPlays = len(State_p.choice_p)
	State_p.estimateReward = [0.0] * numActions

	for i in State_p.choice_p:
		State_p.estimateReward[i] = 1.0 * State_p.scaledReward_p[i] / State_p.probDist_p[i]

	w_temp_p = State_p.weights_pursuer

	for i in range(numActions):
		State_p.weights_pursuer[i] *= math.exp(numPlays * State_p.estimateReward[i] * State_p.gamma_p / numActions) # important that we use estimated reward here!

	for s in State_p.S_0:
		State_p.weights_pursuer[s] = w_temp_p[s]

	# weights_pursuer, estimateReward

def updateState_e(State_e):
	numActions = len(State_e.scaledReward_e)
	State_e.estimatedReward = [0.0] * numActions
	State_e.estimatedReward[State_e.choice_e] = 1.0 * State_e.scaledReward_e[State_e.choice_e] / State_e.probDist_e[State_e.choice_e]
	State_e.weights_evader[State_e.choice_e] *= math.exp(State_e.estimatedReward[State_e.choice_e] * State_e.gamma_e / numActions) # important that we use estimated reward here!



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
		State_p.gamma_p =  min([1, math.sqrt(numActions * math.log(numActions/numPlays) / ((math.e - 1) * numPlays * numRounds ) ) ])

	if config['custom_gamma_evader']:
		State_e.gamma_e =  min([1, math.sqrt(numActions * math.log(numActions) / ((math.e - 1) *  numRounds ) ) ])

	else:
		State_e.gamma_e = config['gamma_e']

	# entering the main loop
	while True:
		theSum_pursuer = sum(State_p.weights_pursuer)
		State_p.weights_pursuer = [w / theSum_pursuer for w in State_p.weights_pursuer] # normalize the weight vector of the pursuer
		theSum_evader = sum(State_e.weights_evader)

		temp = (1.0 / numPlays - State_p.gamma_p / numActions) * float( 1.0 / (1.0 - State_p.gamma_p) )
		w_temp_p = State_p.weights_pursuer
		w_sorted_p = sorted(State_p.weights_pursuer, reverse=True)


		if w_sorted_p[0] >= temp * theSum_pursuer:
			State_p.alpha_t = getAlpha(temp, w_sorted_p)

			State_p.S_0 = find_indices(w_temp_p, lambda e: e >= State_p.alpha_t)

			for s in State_p.S_0:
				w_temp_p[s] = State_p.alpha_t


		else:
			S_0 = []


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

					summary.cumulativeReward_p = 0
					summary.cumulativeRewardVec_p = []
					summary.cumulativeReward_e = 0
					summary.cumulativeRewardVec_e = []

					summary.bestActionCumulativeReward_p = 0
					summary.bestActionCumulativeReward_e = 0

					summary.weakRegret_p = []
					summary.weakRegret_e = []
        			# weakRegret_p = 0
        			# weakRegret_e = 0

					t = 0

					numPlays_LB = nPL_val
					numPlays_UB = nPU_val
					summary.numPlays = [random.randint(numPlays_LB, numPlays_UB) for _ in range(nR_val)] 		# list of number of plays that is uniformly distributed for the pursuer

					try:
						for (State_p, State_e) in advGame(config_search, summary.numPlays[t]):
							summary.cumulativeReward_p += State_p.cumulativeReward_p
							summary.cumulativeRewardVec_p.extend([summary.cumulativeReward_p])
							summary.cumulativeReward_e += State_e.cumulativeReward_e
							summary.cumulativeRewardVec_e.extend([summary.cumulativeReward_e])

							summary.choice_p.append(State_p.choice_p)
							summary.choice_e.append([State_e.choice_e])
							summary.rewardVector_p.append(State_p.rewardFull_p)
							summary.rewardVector_e.append(State_e.rewardFull_e)
                            
							print("pursuer reward:(%s)\tevader reward(%s)\t" % (', '.join(["%.3f" % r for r in State_p.rewardFull_p]),','.join(["%.3f" % r for r in State_e.rewardFull_e])))

							t += 1
							if t>= nR_val:
								break

						summary.bestAction_p = []
						summary.bestAction_e = []

						bestActionSet_p = sorted(range(config_search['numActions']), key=lambda action: sum([summary.rewardVector_p[t][action] for t in range(nR_val)]), reverse=True)[:config_search['numPlays_UB']]
						bestActionSet_e = max(range(config_search['numActions']), key=lambda action: sum([summary.rewardVector_e[t][action] for t in range(nR_val)]))

						for s in range(nR_val):
							summary.bestAction_p.append(bestActionSet_p[summary.numPlays[s]-1])
							summary.bestActionCumulativeReward_p += sum([summary.rewardVector_p[s][i] for i in bestActionSet_p[:summary.numPlays[s]]])
							summary.bestActionCumulativeReward_e += summary.rewardVector_e[s][bestActionSet_e]
        					
							summary.regretBound_p = (1 + (math.e - 2)* config_search['numPlays_UB'] / config_search['numPlays_LB'])* State_p.gamma_p * summary.bestActionCumulativeReward_p + (config_search['numActions'] * math.log(config_search['numActions']/config_search['numPlays_UB'])) / State_p.gamma_p
							summary.regretBound_e = (math.e - 1) * State_e.gamma_e * summary.bestActionCumulativeReward_e + (config_search['numActions'] * math.log(config_search['numActions'])) / State_e.gamma_e

							summary.weakRegret_p.extend([summary.bestActionCumulativeReward_p - summary.cumulativeRewardVec_p[s]])
							summary.weakRegret_e.extend([summary.bestActionCumulativeReward_e - summary.cumulativeRewardVec_e[s]])





					except KeyboardInterrupt:
						pass


