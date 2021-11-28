import numpy as np
import requests

# api endpoint for https://www.wunderground.com/history/monthly/KLGA/date/2020-10
# this has 923 weather data points from LaGuardia airport in NYC over the month of October 2020 under JSON format
url = 'https://api.weather.com/v1/location/KLGA:9:US/observations/historical.json?apiKey=' \
      'e1f10a1e78da46f5b10a1e78da96f525&units=e&startDate=20201001&endDate=20201031'

r = requests.get(url=url).json()  # get raw data from api; this can be found in addendum 1

temps = []
for datapoint in r['observations']:
    # the first data point from each day has the max temperature from that day
    # this checks that there is a value for max_temp and adds it to a list if it is there
    if datapoint['max_temp'] is not None:
        temps.append(int(datapoint['max_temp']))

# here the max temperature from each day is converted into a state, where 0 is 55-59, 1 is 60-64, etc.
# this is done by dividing by five and rounding down then subtracting 11, giving 5 values from 0 to 4
states = []
for i in range(len(temps) - 1):
    states.append(temps[i] // 5 - 11)

# create a 5x5 matrix
matrixSize = 5
transitionStates = np.array(np.arange(matrixSize ** 2), dtype=float).reshape(matrixSize, matrixSize)

# determine the probabilities of going from one state to the next:
# for each combination of starting state and ending state,
# calculate the percentage of times that the starting state
# leads to the ending state and update the corresponding value in the transitionStates array
for stateStart in range(5):
    for stateEnd in range(5):
        start = 0
        end = 0
        for i in range(len(states) - 1):
            if states[i] == stateStart:
                start += 1
                if states[i + 1] == stateEnd:
                    end += 1
        # print(f'{stateStart} to {stateEnd}: {end} / {start} = {round(end / start, 3)}')
        transitionStates[stateStart, stateEnd] = round(end / start, 3)

print(transitionStates)
