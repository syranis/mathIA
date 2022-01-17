import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
import datetime


def get_data(link, state_offset):
    url = link
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
        states.append(temps[i] // 5 - state_offset)
    return states


def simulate_multinomial(vmultinomial):
    r = np.random.uniform(0.0, 1.0)
    cs = np.cumsum(vmultinomial)
    cs = np.insert(cs, 0, 0)
    m = (np.where(cs < r))[0]
    return m[len(m) - 1]


def transition_matrix(states, matrix_size):
    # The following determines the probabilities of going from one state to the next.
    # This is done by iterating over each combination of starting state and ending state to calculate the percentage of
    # times that the starting state leads to the ending state and updating the corresponding value in the
    # transitions array.
    transitions = np.array(np.arange(matrix_size ** 2), dtype=float).reshape(matrix_size, matrix_size)
    for stateStart in range(matrix_size):
        for stateEnd in range(matrix_size):
            start = 0
            end = 0
            for i in range(len(states) - 1):
                if states[i] == stateStart:
                    start += 1
                    if states[i + 1] == stateEnd:
                        end += 1
            transitions[stateStart, stateEnd] = end / start
    print(f'The transition matrix is:\n{transitions}\n')
    return transitions


def stationary_distribution(transitions):
    # The following finds the stationary distribution of the markov chain.
    # The stationary distribution is found here by raising the transition matrix to the power of number of iterations.
    # This works because the stationary distribution is a vector "stationaryDistribution" which when multiplied with the
    # transition matrix returns a vector which can be iteratively multiplied against the transition matrix to reach an
    # equilibrium state.
    # Once two consecutive stationary distributions are equal, the stationary distribution has been calculated.
    distribution = np.zeros((1, np.shape(transitions)[0]), dtype=float)
    distribution[0][0] = 1.0
    state_hist = distribution
    count = 0
    while True:
        if (distribution == np.dot(distribution, transitions)).all():
            print(f'Stationary distribution reached at {count} iterations:\n{distribution}\n')
            break
        count += 1
        distribution = np.dot(distribution, transitions)
        state_hist = np.append(state_hist, distribution, axis=0)
        df_distr_hist = pd.DataFrame(state_hist)
    # noinspection PyUnboundLocalVariable
    df_distr_hist.plot(title='Stationary Distribution')
    plt.show()


def simulate(transitions):
    # The following simulates the markov chain to reconstruct the transition matrix and determine a distribution of
    # temps.
    # By dividing a line segment into intervals proportional to the probabilities in the transition matrix and then
    # generating a uniform random number between 0 and 1, an interval can be chosen.
    # Doing this is a good way to simulate a multinomial distribution and can be applied to markov chains as the
    # collection of moves from any given state form a multinomial distribution.
    state_change_hist = np.array(np.arange(np.shape(transitions)[0] ** 2), dtype=float)\
                        .reshape(np.shape(transitions)[0], np.shape(transitions)[0])
    state_change_hist[state_change_hist > 0.0] = 0.0
    state = np.zeros((1, np.shape(transitions)[0]), dtype=float)
    state[0][0] = 1
    current_state = 0
    state_hist = state

    distr_hist = np.zeros((1, np.shape(transitions)[0]), dtype=int)

    for x in range(50000):
        current_row = np.ma.masked_values((transitions[current_state]), 0.0)
        next_state = simulate_multinomial(current_row)
        # Keep track of state changes
        state_change_hist[current_state, next_state] += 1
        # Keep track of the state vector itself
        state = np.zeros((1, np.shape(transitions)[0]), dtype=int)
        state[0, next_state] = 1.0
        # Keep track of state history
        state_hist = np.append(state_hist, state, axis=0)
        current_state = next_state
        # calculate the actual distribution over the states so far
        totals = np.sum(state_hist, axis=0)
        gt = np.sum(totals)
        distrib = np.reshape(totals / gt, (1, np.shape(transitions)[0]))
        distr_hist = np.append(distr_hist, distrib, axis=0)
    # noinspection PyUnboundLocalVariable
    print(f'Distribution of temperatures:\n{distrib}\n')

    p_hat = state_change_hist / state_change_hist.sum(axis=1)[:, None]
    # Check estimated state transition probabilities based on history so far:
    print(f'Reconstructed transition matrix after 50 000 iterations:\n{p_hat}\n')
    df_distr_hist = pd.DataFrame(distr_hist)
    # Plot the distribution as the simulation progresses over time
    df_distr_hist.plot(title="Simulation History")
    plt.show()


def month(transitions, states):
    # Simulate a month of weather from a starting temperature, in this case, 69, to simulate October 2021
    month_states = []
    current_state = states[0]
    state = np.array([[0.0, 0.0, 0.0, 0.0, 0.0]])
    month_states.append(current_state)
    state[0, current_state] = 1.0
    for i in range(30):
        current_row = np.ma.masked_values((transitions[current_state]), 0.0)
        next_state = simulate_multinomial(current_row)
        state = np.array([[0, 0, 0, 0, 0]])
        state[0, next_state] = 1.0
        month_states.append(next_state)

    correct = 0
    for i in range(len(states)):
        if month_states[i] == states[i]:
            correct += 1
    return correct, len(states)


def main():
    # Data from the api endpoint of https://www.wunderground.com/history/monthly/KLGA/date/2020-10 is being used
    # The data has 923 weather data points from NYC LaGuardia airport over the month of October 2020 under JSON format
    states = get_data('https://api.weather.com/v1/location/KLGA:9:US/observations/historical.json?apiKey='
                      'e1f10a1e78da46f5b10a1e78da96f525&units=e&startDate=20201001&endDate=20201031', 11)
    transitions = transition_matrix(states, 5)

    # Calculate stationary distribution and simulate the chain 50 000 times
    stationary_distribution(transitions)
    simulate(transitions)

    # Find the average accuracy for October 2020 over 100 iterations
    correct, total = (0, 0)
    for i in range(10000):
        accuracy = month(transitions, states)
        correct += accuracy[0]
        total += accuracy[1]
    print(f'Accuracy of {correct / total * 100}% for October 2020')

    # Data from October 2021
    states = get_data('https://api.weather.com/v1/location/KLGA:9:US/observations/historical.json?apiKey='
                      'e1f10a1e78da46f5b10a1e78da96f525&units=e&startDate=20211001&endDate=20211031', 11)
    for i in range(10000):
        accuracy = month(transitions, states)
        correct += accuracy[0]
        total += accuracy[1]
    print(f'Accuracy of {correct / total * 100}% for October 2021')


if __name__ == '__main__':
    main()
