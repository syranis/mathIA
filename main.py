import numpy as np
import pandas as pd
from random import seed
import requests
import matplotlib.pyplot as plt

# Data from the api endpoint of https://www.wunderground.com/history/monthly/KLGA/date/2020-10 is being used
# The data has 923 weather data points from LaGuardia airport in NYC over the month of October 2020 under JSON format
url = 'https://api.weather.com/v1/location/KLGA:9:US/observations/historical.json?apiKey=' \
      'e1f10a1e78da46f5b10a1e78da96f525&units=e&startDate=20201001&endDate=20201031'
r = requests.get(url=url).json()

# The first data point from each day has the max temperature from that day.
# This checks that there is a value for max_temp and adds it to a list if it is there.
temps = []
for datapoint in r['observations']:
    if datapoint['max_temp'] is not None:
        temps.append(int(datapoint['max_temp']))

# The following converts the max temperature from each day into a state, where 0 is 55-59, 1 is 60-64, etc.
# This is done by dividing each value by five and both rounding down and subtracting 11, giving 5 values from 0 to 4
states = []
for i in range(len(temps) - 1):
    states.append(temps[i] // 5 - 11)

# The following determines the probabilities of going from one state to the next.
# This is done by iterating over each combination of starting state and ending state to calculate the percentage of
# times that the starting state leads to the ending state and updating the corresponding value in the transitionMatrix
# array.
matrixSize = 5
transitionMatrix = np.array(np.arange(matrixSize ** 2), dtype=float).reshape(matrixSize, matrixSize)
for stateStart in range(matrixSize):
    for stateEnd in range(matrixSize):
        start = 0
        end = 0
        for i in range(len(states) - 1):
            if states[i] == stateStart:
                start += 1
                if states[i + 1] == stateEnd:
                    end += 1
        transitionMatrix[stateStart, stateEnd] = end / start

print(f'The transition matrix is:\n{transitionMatrix}\n')

# The following finds the stationary distribution of the markov chain.
# The stationary distribution is found here by raising the transition matrix to the power of number of iterations.
# This works because the stationary distribution is a vector "stationaryDistribution" which when multiplied with the
# transition matrix returns a vector which can be iteratively multiplied against the transition matrix to reach an
# equilibrium state.
# Once two consecutive stationary distributions are equal, the stationary distribution has been calculated.
distribution = np.array([[1.0, 0.0, 0.0, 0.0, 0.0]])
stateHist = distribution
dfStateHist = pd.DataFrame(distribution)
count = 0
while True:
    if (distribution == np.dot(distribution, transitionMatrix)).all():
        print(f'Stationary distribution reached at {count} iterations:\n{distribution}')
        break
    count += 1
    distribution = np.dot(distribution, transitionMatrix)
    stateHist = np.append(stateHist, distribution, axis=0)
    dfDistrHist = pd.DataFrame(stateHist)
dfDistrHist.plot(title='Stationary Distribution')
plt.show()

# The following simulates the markov chain.
# By dividing a line segment into intervals proportional to the probabilities in the transition matrix and then
# generating a uniform random number between 0 and 1, an interval can be chosen.
# Doing this is a good way to simulate a multinomial distribution and can be applied to markov chains as the collection
# of moves from any given state form a multinomial distribution.
stateChangeHist = np.array(np.arange(matrixSize ** 2), dtype=float).reshape(matrixSize, matrixSize)
stateChangeHist[stateChangeHist > 0.0] = 0.0
state = np.array([[1.0, 0.0, 0.0, 0.0, 0.0]])
currentState = 0
stateHist = state
dfStateHist = pd.DataFrame(state)

distr_hist = [[0, 0, 0, 0, 0]]
seed(4)


def simulate_multinomial(vmultinomial):
    r = np.random.uniform(0.0, 1.0)
    CS = np.cumsum(vmultinomial)
    CS = np.insert(CS, 0, 0)
    m = (np.where(CS < r))[0]
    nextState = m[len(m) - 1]
    return nextState


for x in range(50000):
    currentRow = np.ma.masked_values((transitionMatrix[currentState]), 0.0)
    nextState = simulate_multinomial(currentRow)
    # Keep track of state changes
    stateChangeHist[currentState, nextState] += 1
    # Keep track of the state vector itself
    state = np.array([[0, 0, 0, 0, 0]])
    state[0, nextState] = 1.0
    # Keep track of state history
    stateHist = np.append(stateHist, state, axis=0)
    currentState = nextState
    # calculate the actual distribution over the 5 states so far
    totals = np.sum(stateHist, axis=0)
    gt = np.sum(totals)
    distrib = np.reshape(totals / gt, (1, 5))
    distr_hist = np.append(distr_hist, distrib, axis=0)
print(f'Distribution of temperatures:\n{distrib}\n')

P_hat = stateChangeHist / stateChangeHist.sum(axis=1)[:, None]
# Check estimated state transition probabilities based on history so far:
print(f'Reconstructed transition matrix after 50 000 iterations:\n{P_hat}\n')
dfDistrHist = pd.DataFrame(distr_hist)
# Plot the distribution as the simulation progresses over time
dfDistrHist.plot(title="Simulation History")
plt.show()
