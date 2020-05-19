from probability import distr_multiPlays
from DepRound import DepRound, DepRound1
import math
import random
# from sympy.solvers import solve
# from sympy import Symbol
from scipy.optimize import fsolve
import numpy as np

# perform the Exp3.M algorithm, which is a mutli-play version of Exp3;
# numActions: number of locations, indexed from 0;
# reward_multiPlays: function accepting as input the multiple locations and producing as output the reward vector for the location set;
# gamma: egalitarianism factor;
# numPlays: number of arms played at each time

def getAlpha(temp, w_sorted):
	# getAlpha calculates the alpha value for the sorted weight.
	sum_weight = sum(w_sorted)

	for i in range(len(w_sorted)):
		alpha = (temp * sum_weight) / (1 - i * temp)
		curr = w_sorted[i]

		if alpha > curr:
			alpha_exp = alpha
			return alpha_exp

		sum_weight = [s - curr for s in sum_weight]
	raise Exception('alpha not found')


def exp3m(numActions,reward_multiPlays,gamma,numPlays,rewardMin = 0, rewardMax = 1):
	weights = [1.0] * numActions # initialize weight vector

	t = 0
	while True:
		theSum = sum(weights)
		weights = [w / theSum for w in weights] # normalize the weight vector
		temp =  (1.0 / numPlays - gamma / numActions) * float( 1.0 / (1.0 - gamma) ) 
		w_temp1 = weights

		if max(weights) >= temp * theSum:
			# alpha = Symbol('alpha')
			# w_temp = weights
			# w_temp.sort() # sort weights vector in asceding order, optional

			# fun_1 = lambda alpha: alpha / (sum( [alpha for w in weights if w >= alpha] ) + sum( [w for w in weights if w < alpha] ) ) - (1.0 / numPlays - gamma / numActions) / (1.0 - gamma)

			# x_initial = 0													# set initial search point for fsolve solver to find the value alpha_t
			# alpha_t = fsolve(fun_1, x_initial)
			w_sorted = sorted(weights, reverse=True)
			alpha_t = getAlpha(temp,w_sorted)
			# alpha_t = alpha_t.tolist() # convert numpy output by fsolve solver to list

			idx_temp = 0
			S_null = []

			
			for w in weights:
				if w >= alpha_t:
					S_null.append(idx_temp)
				idx_temp += 1

			for s in S_null:
				w_temp1[s] = alpha_t

		else:
			S_null = []


		probabilityDistribution = distr_multiPlays(w_temp1,numPlays,gamma = gamma)
		# if True in np.isnan(np.array(probabilityDistribution)):
		# 	pass

		assert False in np.isnan(np.array(probabilityDistribution)), "Error, probability must be a real number"

		choice = sorted(DepRound(probabilityDistribution,k = numPlays))				# list of choice
		# choice = DepRound1(probabilityDistribution,k = numPlays)					# list of choice
		theReward = reward_multiPlays(t, choice)							# input: choice list and time; output: reward list
		

		theReward_full = [0.0] * numActions # initialize reward vector

		for i in choice:
			for r in theReward:
				theReward_full[i] = r
																			
		

		scaledReward = [(r - rewardMin) / (rewardMax - rewardMin) for r in theReward_full] # reward vector scaled to 0,1, optional
		# probabilityChoice = [probabilityDistribution[i] for i in choice] 			# probability vector of selected locations

		estimateReward = [0.0] * numActions

		for i in choice:
			estimateReward[i] = 1.0 * scaledReward[i] / probabilityDistribution[i] 
		
		w_temp = weights
        
		for i in range(numActions):
			weights[i] *= math.exp(numPlays * estimateReward[i] * gamma / numActions) # important that we use estimated reward here!
		


		for s in S_null:
			weights[s] = w_temp[s]


		cumulativeReward = sum(theReward)

		if (sum(weights) == 0):
			pass 					# Debugging

		yield choice, cumulativeReward, estimateReward, weights
		t = t + 1



# Test Exp3.M using stochastic payoffs for 10 actions with 3 plays at each time.
def simpleTest():
   numActions = 10
   numRounds = 200000
   numPlays = 2

   biases = [1.5 / k for k in range(2,2 + numActions)]
   rewardVector = [[1 if random.random() < bias else 0 for bias in biases] for _ in range(numRounds)]

   rewards = lambda t, choice: [rewardVector[t][i] for i in choice] 
   # if (numPlays > 1):
   		# rewards = lambda t, choice: [rewardVector[t][i] for i in choice] 

   # else:
   # 		rewards = lambda t, choice: rewardVector[t][choice]

   bestAction = sorted(range(numActions), key=lambda action: sum([rewardVector[t][action] for t in range(numRounds)]), reverse=True)[:numPlays]
   

   
   gamma = min([1, math.sqrt(numActions * math.log(numActions/numPlays) / ((math.e - 1) * numPlays * numRounds ) ) ])
   gamma = 0.07

   cumulativeReward = 0
   bestActionCumulativeReward = 0
   weakRegret = 0

   t = 0
   for (choice, reward, est, weights) in exp3m(numActions, rewards, gamma,numPlays):
      cumulativeReward += reward
      bestActionCumulativeReward += sum([rewardVector[t][i] for i in bestAction])
      bestUpperBoundEstimate = (math.e - 1) * gamma * bestActionCumulativeReward

      weakRegret = (bestActionCumulativeReward - cumulativeReward)
      averageRegret = weakRegret / (t+1)

      # regretBound = (math.e - 1) * gamma * bestActionCumulativeReward + (numActions * math.log(numActions)) / gamma
      # regretBound = 2.63 * math.sqrt(numActions * t * numPlays * math.log(numActions/numPlays) )

      regretBound = bestUpperBoundEstimate + (numActions * math.log(numActions / numPlays)) / gamma

      print("regret: %d\tmaxRegret: %.2f\taverageRegret: %.2f\tweights: (%s)" % (weakRegret, regretBound, averageRegret, ', '.join(["%.3f" % weight for weight in distr_multiPlays(weights,numPlays)])))




      t += 1


      if t >= numRounds:
         break

   print(cumulativeReward)


if __name__ == "__main__":
   simpleTest()
		



