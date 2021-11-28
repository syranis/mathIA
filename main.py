import math
import numpy as np
import requests

# api for https://www.wunderground.com/history/monthly/KLGA/date/2020-10
url = 'https://api.weather.com/v1/location/KLGA:9:US/observations/historical.json?apiKey=' \
      'e1f10a1e78da46f5b10a1e78da96f525&units=e&startDate=20201001&endDate=20201031'

r = requests.get(url=url).json()  # get raw data from api; this can be found in addendum 1

temps = []
for datapoint in r['observations']:
    # the first data point from each datapoint has the max temperature from that day
    # this checks that there is a value for max_temp and adds it to a list
    if datapoint['max_temp'] is not None:
        temps.append(int(datapoint['max_temp']))


# here the max temperature from each day is converted into a state, where 0 is 45F-49F, 1 is 50-54, etc.
state = []
for i in range(len(temps)):
    state.append(math.floor(temps[i] / 5) - 9)

# for i in range(len(state)):
#     print(f'{i} | {state[i]}')

transitionStates = np.array(range(0, 49)).reshape((7, 7))

print(state)
print(transitionStates)

# determine the probabilities of going from one state to the next
for i in range(len(state)):
    for start in range(7):
        continue
