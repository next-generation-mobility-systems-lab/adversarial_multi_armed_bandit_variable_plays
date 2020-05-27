from probability import distr_multiPlays
import math
import random
import numpy as np
import matplotlib.pyplot as plt

from exp3M import exp3m
from utilities import randomInt

# perform the Exp3.M algorithm, which is a mutli-play version of Exp3;
# numActions: number of locations, indexed from 0;
# reward_multiPlays: function accepting as input the multiple locations and producing as output the reward vector for the location set;
# gamma: egalitarianism factor;
# numPlays: number of arms played at each time



# Test Exp3.M using stochastic payoffs for 10 actions with 3 plays at each time.
def simpleTest():
   numActions = 10
   numRounds = 200000
   numPlays_LB = 1
   numPlays_UB = 3
   numPlays_std = 0.8

   numPlays = randomInt(numPlays_LB, numPlays_UB, numPlays_std, numRounds)


   biases = [1.5 / k for k in range(2,2 + numActions)]
   rewardVector = [[1 if random.random() < bias else 0 for bias in biases] for _ in range(numRounds)]

   rewards = lambda t, choice: [rewardVector[t][i] for i in choice]
   # if (numPlays > 1):
   		# rewards = lambda t, choice: [rewardVector[t][i] for i in choice]

   # else:
   # 		rewards = lambda t, choice: rewardVector[t][choice]

   bestActionSet = sorted(range(numActions), key=lambda action: sum([rewardVector[t][action] for t in range(numRounds)]), reverse=True)[:numPlays_UB]



   gamma = min([1, math.sqrt(numActions * numActions * numPlays_LB *math.log(numActions/numPlays_UB)\
			 / (( (math.e - 2) * numPlays_UB + numPlays_LB)* numPlays_UB * numRounds ) ) ])
   gamma = 0.05

   cumulativeReward = 0
   bestActionCumulativeReward = 0
   weakRegret = 0
   weakRegretVec=[]
   linear_regret = []
   regret_Bound_vec = []
   weights_vec = []
   factor = []
   average_reward = []

   t = 0
   for (choice, reward, est, weights) in exp3m(numActions, rewards, gamma, numPlays[t]):
      cumulativeReward += reward
      average_reward.extend([cumulativeReward / (t+1)])
      bestActionCumulativeReward += sum([rewardVector[t][i] for i in bestActionSet[:numPlays[t]]])
      # bestUpperBoundEstimate = (math.e - 1) * gamma * bestActionCumulativeReward

      weakRegret = (bestActionCumulativeReward - cumulativeReward)
      averageRegret = weakRegret / (t+1)
      weights_vec.append(distr_multiPlays(weights,1))


      # regretBound = (math.e - 1) * gamma * bestActionCumulativeReward + (numActions * math.log(numActions)) / gamma
      # regretBound = 2.63 * math.sqrt(numActions * t * numPlays * math.log(numActions/numPlays) )

      # regretBound = bestUpperBoundEstimate + (numActions * math.log(numActions / numPlays)) / gamma
      regretBound = (1 + (math.e - 2)* numPlays_UB / numPlays_LB)* gamma \
								* bestActionCumulativeReward + (numActions * math.log(numActions/numPlays_UB)) / gamma


      factor.append(weakRegret/regretBound)

      weakRegretVec.append(weakRegret)
      linear_regret.append(t+1)
      regret_Bound_vec.append(regretBound)

      print("regret: %d\tmaxRegret: %.2f\taverageRegret: %.2f\tweights: (%s)" % (weakRegret, regretBound, averageRegret, ', '.join(["%.3f" % weight for weight in distr_multiPlays(weights,numPlays[t])])))




      t += 1


      if t >= numRounds:
         break

   print(cumulativeReward)

   # plotting
   fig1, (ax1,ax2) = plt.subplots(nrows=2, ncols=1)
   plt.ylabel('Cumulative (weak) Regret')
   ax1.plot(range(numRounds), weakRegretVec,label='weak regret')
   ax1.plot(range(numRounds), linear_regret,label='linear regret')
   ax1.plot(range(numRounds), regret_Bound_vec, label = 'expected upper bound')
   ax1.legend()

   np_weights_vec = np.array(weights_vec)
   transpose = np_weights_vec.T
   weights_vec = transpose.tolist()

   for w in weights_vec:
       ax2.plot(range(numRounds), w)

   plt.ylabel('Weight')


   fig2, (ax3,ax4) = plt.subplots(nrows=2, ncols=1)
   ax3.plot(range(numRounds), factor, label = 'weak regret/upper bound')
   ax3.set_title("weak regret/upper bound")

   ax4.plot(range(numRounds), average_reward, label = 'average reward')
   ax4.set_title("average reward")








if __name__ == "__main__":
   simpleTest()




