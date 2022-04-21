# Simulated annealing
#
# Using a threshold accepting simulated annealing algorithm
# And a hill climbing algorithm
# for the traveling salesman problem with 100 cities

# Questions!
# With algorithm performs better?
# What algorithmic choices did you make?

# Presentations General Guidelines
# I.    Motivation
# II.   Problem Statement
# III.  Related Work (only what is needed to discuss the contributions)
# IV.   Contributions
# V.    Related Work and Background Material (in Detail)
# VI.   Approach
# VII.  Experiments
# VIII. Results
# IX.   Conclusions


# cooling schedules?
import math

import numpy as np
import matplotlib.pyplot as plt  # plot results

# np.random.seed(42)


def random_coordinates():
    # _coordinates = np.random.randint(low=0, high=20, size=2)
    # return [_coordinates[0], _coordinates[1]]
    return [np.random.random(), np.random.random()]


def route_cost(route_list):
    """Calculate route cost from the root of the list to the end of the list"""
    _distance_sum = 0
    # sqrt[(x2-x1)^2 + (y2-y1)^2]
    _length = len(route_list)
    for i in range(_length):
        if i + 1 < _length:
            """Distance between current node and next node"""
            x1, y1 = route_list[i]
            x2, y2 = route_list[i + 1]
        else:
            """When we reach the last node complete the circle
                Sum current node with the first node"""
            x1, y1 = route_list[i]  # tail
            x2, y2 = route_list[0]  # first node

        _sum = (x2 - x1) ** 2 + (y2 - y1) ** 2
        _distance_sum += math.sqrt(_sum)
    return _distance_sum


def neighbor_swap(cities):
    _r1, _r2 = np.random.randint(0, len(cities), size=2)
    # swap
    _tmp = cities[_r1]
    cities[_r1] = cities[_r2]
    cities[_r2] = _tmp
    return cities


class Cities:
    """City Generation testing"""

    def __init__(self, quantity):
        self.cities = []
        self.total_cities = quantity

    def generate(self):
        while len(self.cities) < self.total_cities:
            _coordinates = random_coordinates()
            _gx, _gy = _coordinates[0], _coordinates[1]
            if [_gx, _gy] in self.cities:
                print("Dupe entry -- regenerating coordinates")
                continue
            else:
                self.cities.append([_gx, _gy])
        self.cities.append([self.cities[0][0], self.cities[0][1]])
        return self.cities


class SATAV:
    """Simulated Annealing Threshold Accepting Variant"""

    def __init__(self, cities):
        self.cities = cities

    def simulate(self, rounds):  # TA Variant 00:34:10
        """
        The threshold sequence is analogous to the cooling schedule
        Threshold is analogous to temperature
        """
        tau = self.generate_threshold_vector(rounds)
        s = self.cities.copy()
        s_new = s.copy()
        best = route_cost(s)
        for rnd in range(rounds):
            for t in tau:
                # Neighbor function is what makes nice solutions
                # this is the algorithm talking point of what makes this 00:14:40
                s_new = neighbor_swap(s_new)  # neighbor(s) #For every temperature in every round calculate a new state
                _cost = route_cost(s_new) - route_cost(s)
                # print("Cost", _cost)

                if _cost > best:
                    continue

                if _cost < best and _cost < t:
                    s = s_new.copy()
                    best = _cost

        # print(max(_route_diff), min(_route_diff))
        return s

    # def generate_threshold_vector(self, rounds):
    #     _threshold_vector = []
    #     _city_list = self.cities.copy()
    #     s_new = self.cities.copy()
    #     for rnd in range(rounds):
    #         # Run through the rounds and capture all route costs with random swaps
    #         # For every swap calculate a threshold
    #         s_new = neighbor_swap(s_new)
    #         if route_cost(s_new) < route_cost(_city_list):
    #             _city_list = s_new.copy()
    #             _threshold_vector.append(route_cost(s_new) - route_cost(_city_list))
    #
    #     # Reverse sort list to apply large change first
    #     _threshold_vector.sort(reverse=True)
    #     print("Length of vector:", len(_threshold_vector))
    #     return _threshold_vector

    def generate_threshold_vector(self, rounds):
        _threshold_vector = []
        init_temp = 1
        final_temp = route_cost(self.cities)
        t = init_temp
        for k in range(2, rounds):
            b = (init_temp - final_temp) / ((k - 1) * init_temp * final_temp)
            t = t / (1 + b * t)
            _threshold_vector.append(t)
        print("Threshold vector:", len(_threshold_vector))
        print("Iitial", init_temp)
        print("First iter", _threshold_vector[0])
        return _threshold_vector


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    city = Cities(100)
    city_list = city.generate()
    # print("City List:", city_list)
    print("Starting route cost:", route_cost(city_list))

    # plot the cities and paths
    fig = plt.figure(figsize=(10, 5))
    _x, _y = [], []
    for x, y in city_list:
        plt.scatter(x, y)
        _x.append(x)
        _y.append(y)
    plt.title("City Scatter Plot")
    plt.plot(_x, _y)
    plt.show()
    print(_x, _y)

    sa_variant = SATAV(city_list)  # Simulated annealing threshold accepting variant
    new_city_list = sa_variant.simulate(2_000)

    fig2 = plt.figure(figsize=(10, 5))

    _x, _y = [], []
    for x, y in new_city_list:
        plt.scatter(x, y)
        _x.append(x)
        _y.append(y)
    # Connect head to tail
    _x.append(new_city_list[0][0])
    _y.append(new_city_list[0][1])
    plt.plot(_x, _y)
    plt.show()
    print(_x, _y)

    print("Final route cost:", (route_cost(new_city_list)))
