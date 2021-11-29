import numpy as np
import pandas as pd
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

print(f'The transition matrix is:\n{transitionMatrix}')

# The following finds the stationary distribution of the markov chain.
# The stationary distribution is found here by raising the transition matrix to the power of number of iterations.
# This works because the stationary distribution is a vector "stationaryDistribution" which when multiplied with the
# transition matrix returns a vector which can be iteratively multiplied against the transition matrix to reach an
# equilibrium state.
# Once two consecutive stationary distributions are equal, the stationary distribution has been calculated.
distribution = np.array([[1.0, 0.0, 0.0, 0.0, 0.0]])
stateHist = distribution
dfStateHist = pd.DataFrame(distribution)
distr_hist = [[0, 0, 0]]
count = 0
while True:
    if (distribution == np.dot(distribution, transitionMatrix)).all():
        print(f'Stationary distribution reached at {count} iterations:\n{distribution}')
        break
    count += 1
    distribution = np.dot(distribution, transitionMatrix)
    stateHist = np.append(stateHist, distribution, axis=0)
    dfDistrHist = pd.DataFrame(stateHist)
    dfDistrHist.plot()
plt.show()

# The following simulates the markov chain.
