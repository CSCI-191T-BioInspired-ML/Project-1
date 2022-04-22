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

import math
import time

import numpy as np
import matplotlib.pyplot as plt  # plot results


def random_coordinates():
    return [np.random.random(), np.random.random()]


def route_cost(route_list):
    """Calculate route cost from the root of the list to the end of the list"""
    _distance_sum = 0
    # sqrt[(x2-x1)^2 + (y2-y1)^2]
    _length = len(route_list)
    for i in range(_length - 1):
        """Distance between current node and next node"""
        x1, y1 = route_list[i]
        x2, y2 = route_list[i + 1]
        _sum = (x2 - x1) ** 2 + (y2 - y1) ** 2
        _distance_sum += math.sqrt(_sum)
    """Last node to the root node"""
    x1, y1 = route_list[0]
    x2, y2 = route_list[len(route_list) - 1]
    _sum = (x2 - x1) ** 2 + (y2 - y1) ** 2
    _distance_sum += math.sqrt(_sum)
    return _distance_sum


def neighbor_swap(cities):
    in_a, in_b = np.random.randint(0, len(cities), size=2)
    while in_a == in_b:  # don't swap self
        in_a, in_b = np.random.randint(0, len(cities), size=2)
    cities[in_a], cities[in_b] = cities[in_b], cities[in_a]
    return cities


class Cities:
    """City list generator"""

    def __init__(self, quantity):
        self.cities = []
        self.total_cities = quantity

    def generate(self):
        self.cities.clear()
        while len(self.cities) < self.total_cities:
            _coordinates = random_coordinates()
            _gx, _gy = _coordinates[0], _coordinates[1]
            if [_gx, _gy] in self.cities:
                print("Dupe entry -- regenerating coordinates")
                continue
            else:
                self.cities.append([_gx, _gy])
        return self.cities


class SimulatedAnnealingTA:
    """Simulated Annealing Threshold Accepting Variant"""

    def __init__(self, cities):
        self.cities = cities
        self.best_route = cities

    def simulate(self, rounds, steps):
        """
        The threshold sequence is analogous to the cooling schedule (temperature)
        """
        tau = self.generate_threshold_vector_quad(steps)  # length of threshold vector
        s_old = self.cities.copy()
        # s_new = s_old.copy()
        for _ in range(rounds):
            s_new = self.best_route.copy()
            for t in tau:
                s_new = neighbor_swap(s_new)  # swap neighbors for a new state
                _cost = route_cost(s_new) - route_cost(s_old)
                if _cost < t:
                    s_old = s_new.copy()
                    if route_cost(s_old) < route_cost(self.best_route):
                        self.best_route = s_new.copy()
        return self.best_route

    def generate_threshold_vector_exp(self, rounds):
        _threshold_vector = []
        t0 = 100
        k = 0.8
        for r in range(rounds):
            t = t0 * k ** r
            _threshold_vector.append(t)
        # print("TA vector list (Exponential)", _threshold_vector)
        return _threshold_vector

    def generate_threshold_vector_quad(self, rounds):
        _threshold_vector = []
        t0 = 100
        k = 0.2
        for r in range(rounds):
            t = t0 / (1 + k * (r ** 2))
            _threshold_vector.append(t)
        # print("TA vector list (Quadratic multiplicative)", _threshold_vector)
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
    _x.append(city_list[0][0])
    _y.append(city_list[0][1])
    plt.title("City Scatter Plot")
    plt.plot(_x, _y)
    plt.show()
    print(_x, _y)

    sata = SimulatedAnnealingTA(city_list)  # Simulated annealing threshold accepting variant
    new_city_list = sata.simulate(250, 100)  # rounds, threshold length

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
    print("\t\tDIFF", (route_cost(city_list) - route_cost(new_city_list)))

    # Testing
    # city = Cities(100)
    # city_list = city.generate()
    # for i in range(1, 11):
    #     start = time.time()
    #     print("Testing", (100 * i), "rounds with 100 threshold vector length")
    #     print("\tStarting route cost:", route_cost(city_list))
    #     sata = SimulatedAnnealingTA(city_list)  # Simulated annealing threshold accepting variant
    #     new_city_list = sata.simulate(100 * i, 100)  # rounds, threshold length
    #     print("\tFinal route cost:", (route_cost(new_city_list)), "Time: ", (time.time() - start))
    #     print("\t\tDIFF Rounds", (route_cost(city_list) - route_cost(new_city_list)))
    #     start = time.time()
    #     print("Testing 100 rounds with", (100 * i), "threshold vector length")
    #     print("\tStarting route cost:", route_cost(city_list))
    #     sata = SimulatedAnnealingTA(city_list)  # Simulated annealing threshold accepting variant
    #     new_city_list = sata.simulate(100, 100 * i)  # rounds, threshold length
    #     print("\tFinal route cost:", (route_cost(new_city_list)), "Time: ", (time.time() - start))
    #     print("\t\tDIFF Vector", (route_cost(city_list) - route_cost(new_city_list)))
    #     start = time.time()
    #     print("Testing", (100//2 * i), "rounds with", (100//2 * i), "threshold vector length")
    #     print("\tStarting route cost:", route_cost(city_list))
    #     sata = SimulatedAnnealingTA(city_list)  # Simulated annealing threshold accepting variant
    #     new_city_list = sata.simulate(100//2 * i, 100//2 * i)  # rounds, threshold length
    #     print("\tFinal route cost:", (route_cost(new_city_list)), "Time: ", (time.time() - start))
    #     print("\t\tDIFF Vector", (route_cost(city_list) - route_cost(new_city_list)))
    #     print("-------------------------------------")
