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
from enum import Enum

import numpy as np
import matplotlib.pyplot as plt  # plot results
from numpy.random import default_rng

rng = default_rng(seed=420)
benchmark_seconds = 60


def random_coordinates():
    return [rng.random(), rng.random()]


def route_cost(route_list):
    """Calculate route cost from the root of the list to the end of the list"""
    _distance_sum = 0
    # sqrt[(x2-x1)^2 + (y2-y1)^2]
    _length = len(route_list)
    for coord in range(_length - 1):
        """Distance between current node and next node"""
        x1, y1 = route_list[coord]
        x2, y2 = route_list[coord + 1]
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


class CoolingSchedule(Enum):
    quad_multiplicative = "Quadratic"
    exponential = "Exponential"


class SimulatedAnnealingTA:
    """Simulated Annealing Threshold Accepting Variant"""

    def __init__(self, cities, schedule: CoolingSchedule, timeout_seconds):
        self.cities = cities
        self.best_route = cities
        # Only run the simulation for a certain amount of time (s)
        self.timeout = timeout_seconds
        # Use nanoseconds for extra precision--convert seconds to ns by multiplying 1e+9 (nano)
        self.timeout_ns = time.time_ns() + (timeout_seconds * 1e+9)
        self.time_spent = 0
        # Accept different cooling schedules for the generation of the threshold vector
        self.schedule = schedule

    def simulate(self, rounds, steps, timeout: bool):
        """
        The threshold sequence is analogous to the cooling schedule (temperature)
        """
        # Generate threshold vector with the assigned cooling schedule
        if self.schedule == self.schedule.exponential:
            tau = self.generate_threshold_vector_exp(steps)
        else:
            tau = self.generate_threshold_vector_quad(steps)
        s_old = self.cities.copy()

        if timeout:
            print("Running TA for", self.get_bench_time_seconds(), "seconds...")
            while time.time_ns() < self.timeout_ns:
                s_new = self.best_route.copy()
                for t in tau:
                    s_new = neighbor_swap(s_new)  # swap neighbors for a new state
                    _cost = route_cost(s_new) - route_cost(s_old)
                    if _cost < t:
                        s_old = s_new.copy()
                        if route_cost(s_old) < route_cost(self.best_route):
                            self.best_route = s_new.copy()
                    if time.time_ns() > self.timeout_ns:
                        finish_time = self.get_runtime_ms()
                        self.update_time_spent_ms(finish_time)
                        # print("Stopping benchmark it we are", finish_time, "ms overdue")
                        break
        else:
            print(f'Running TA for {rounds} rounds using a threshold vector size of {len(tau)}...')
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

    def get_runtime_ms(self):
        # 1 ms = 1,000,000 ns
        return (time.time_ns() - self.timeout_ns) / 1e+6

    def get_bench_time_seconds(self):
        # 1 s = 1,000,000,000 ns
        return self.timeout

    def update_time_spent_ms(self, spent_time_ms):
        self.time_spent = spent_time_ms

    def get_time_spent_ms(self):
        return self.time_spent

    @staticmethod
    def generate_threshold_vector_exp(rounds):
        _threshold_vector = []
        t0 = 100
        k = 0.8
        for r in range(rounds):
            t = t0 * k ** r
            _threshold_vector.append(t)
        print("\tgenerated an exponential threshold vector with a length of", rounds)
        print(f'\t\tvector list preview -> [{_threshold_vector[0]}, {_threshold_vector[1]}, {_threshold_vector[2]},'
              f' ..., {_threshold_vector[-1]}]')
        return _threshold_vector

    @staticmethod
    def generate_threshold_vector_quad(rounds):
        _threshold_vector = []
        t0 = 100
        k = 0.2
        for r in range(rounds):
            t = t0 / (1 + k * (r ** 2))
            _threshold_vector.append(t)
        print("\tgenerated a quadratic multiplicative threshold vector with a length of", rounds)
        print(f'\t\tvector list preview -> [{_threshold_vector[0]}, {_threshold_vector[1]}, {_threshold_vector[2]},'
              f' ..., {_threshold_vector[-1]}]')
        return _threshold_vector


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    city = Cities(100)
    city_list = city.generate()
    # print("City List:", city_list)
    print("Starting route cost:", route_cost(city_list))

    # plot the cities and paths
    # fig = plt.figure(figsize=(10, 5))
    # _x, _y = [], []
    # for x, y in city_list:
    #     plt.scatter(x, y)
    #     _x.append(x)
    #     _y.append(y)
    # _x.append(city_list[0][0])
    # _y.append(city_list[0][1])
    # plt.title("City Scatter Plot")
    # plt.plot(_x, _y)
    # plt.show()
    # print(_x, _y)

    # Simulated annealing threshold accepting variant
    sata = SimulatedAnnealingTA(city_list, CoolingSchedule.quad_multiplicative, benchmark_seconds)
    new_city_list = sata.simulate(2000, 400, True)  # rounds, threshold length, use benchmark instead of rounds

    # fig2 = plt.figure(figsize=(10, 5))

    # _x, _y = [], []
    # for x, y in new_city_list:
    #     plt.scatter(x, y)
    #     _x.append(x)
    #     _y.append(y)
    # # Connect head to tail
    # _x.append(new_city_list[0][0])
    # _y.append(new_city_list[0][1])
    # plt.plot(_x, _y)
    # plt.show()
    # print(_x, _y)
    print("\tBenchmark of", sata.get_bench_time_seconds(), "seconds went over: ", sata.get_time_spent_ms(), "ms")
    print("Final route cost:", (route_cost(new_city_list)))
    print(f'\t{route_cost(city_list)} -> {route_cost(new_city_list)}')
    print("\t\tcost improvement difference", (route_cost(city_list) - route_cost(new_city_list)))

    # # Testing
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
    #     print("Testing", (100 // 2 * i), "rounds with", (100 // 2 * i), "threshold vector length")
    #     print("\tStarting route cost:", route_cost(city_list))
    #     sata = SimulatedAnnealingTA(city_list)  # Simulated annealing threshold accepting variant
    #     new_city_list = sata.simulate(100 // 2 * i, 100 // 2 * i)  # rounds, threshold length
    #     print("\tFinal route cost:", (route_cost(new_city_list)), "Time: ", (time.time() - start))
    #     print("\t\tDIFF Vector", (route_cost(city_list) - route_cost(new_city_list)))
    #     print("-------------------------------------")
