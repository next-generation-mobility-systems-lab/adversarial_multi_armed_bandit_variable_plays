from probability import distr, distr_multiPlays, draw
from DepRound import DepRound
import math
import random
from sympy.solvers import solve
from sympy import Symbol


# perform the Exp3.M algorithm, which is a mutli-play version of Exp3;
# numActions: number of locations, indexed from 0;
# reward_multiPlays: function accepting as input the multiple locations and producing as output the reward vector for the location set;
# gamma: egalitarianism factor;
# numPlays: number of arms played at each time

def exp3m(numActions,reward_multiPlays,gamma,numPlays,rewardMin = 0, rewardMax = 1)
	weights = [1.0] * numActions # initialize weight vector

	t = 0
	while True:
		temp =  (1 / numPlays - gamma / numActions) * float( sum(weights) / (1.0 - gamma) ) 

		if (w >= temp for w in weights):
			alpha = Symbol('alpha')
			w_temp = weights
			# w_temp.sort() # sort weights vector in asceding order, optional


			alpha_t = solve( alpha / (sum( [w_temp for w in w_temp if w >= alpha] ) + ...
				sum( [w_temp for w in w_temp if w < alpha] ) ) - ...
				 (1 / numPlays - gamma / numActions) / (1.0 - gamma), alpha)

			idx_temp = 0
			S_null = []

			for w in w_temp:
				if w >= alpha_t:
					S_null = S_null.append(idx_temp)
				idx_temp += 1


			for s in S_null:
				w_temp[s] = alpha_t

		else
			S_null = []


		probabilityDistribution = distr_multiPlays(w_temp,gamma,numPlays)
		choice = DepRound(probabilityDistribution,numPlays)
		theReward = reward_multiPlays(choice, t)
		


