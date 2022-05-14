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
benchmark_seconds = 90


def random_coordinates():
    return [rng.random(), rng.random()]


def single_dist(_a, _b):
    return math.sqrt((_a[0] - _b[0]) ** 2 + (_a[1] - _b[1]) ** 2)


def get_total_dist(_city_list):
    dist = 0
    # create tuples, everything except last item and last item
    for _a, _b in zip(_city_list[:-1], _city_list[1:]):
        dist += single_dist(_a, _b)
    dist += single_dist(_city_list[0], _city_list[-1])
    return dist


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
    quad_multiplicative = "Quadratic Multiplicative"
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
        loops_counter = 0
        update_counter = 0
        s_old_cost = get_total_dist(s_old)
        s_best_cost = get_total_dist(self.best_route)

        if timeout:
            print("Running TA for", self.get_bench_time_seconds(), "seconds...")
            while time.time_ns() < self.timeout_ns:
                s_new = self.best_route.copy()
                for t in tau:
                    """Check for improvement before visiting the next threshold"""
                    improvement = 0
                    while improvement > -2:  # evaluate improvement of objective function
                        s_new = neighbor_swap(s_new)  # swap neighbors for a new state
                        s_new_cost = get_total_dist(s_new)  # cost of new route
                        _cost = s_new_cost - s_old_cost

                        """Keep track of the objective function improvement"""
                        if s_new_cost > s_old_cost:
                            improvement -= 1  # objective function is not improving at this temperature

                        """Test the threshold temperature"""
                        if _cost < t:
                            # accept the threshold
                            s_old = s_new.copy()
                            s_old_cost = get_total_dist(s_old)

                            """Update best solution"""
                            if s_old_cost < s_best_cost:  # update the best route
                                update_counter += 1
                                self.best_route = s_old.copy()  # update best state of the traveling salesman
                                s_best_cost = get_total_dist(self.best_route)

                        if time.time_ns() > self.timeout_ns:  # exit while loop
                            finish_time = self.get_runtime_ms()
                            self.update_time_spent_ms(finish_time)
                            # print("Stopping benchmark it we are", finish_time, "ms overdue")
                            break

                        loops_counter += 1
                    if time.time_ns() > self.timeout_ns:
                        break
                if time.time_ns() > self.timeout_ns:  # exit for loop
                    break
        else:
            print(f'Running TA for {rounds} rounds using a threshold vector size of {len(tau)}...')
            for _ in range(rounds):
                s_new = self.best_route.copy()
                for t in tau:
                    """Check for improvement before visiting the next threshold"""
                    improvement = 0
                    while improvement > -2:  # evaluate improvement of objective function
                        s_new = neighbor_swap(s_new)  # swap neighbors for a new state
                        s_new_cost = get_total_dist(s_new)  # cost of new route
                        _cost = s_new_cost - s_old_cost

                        """Keep track of the objective function improvement"""
                        if s_new_cost > s_old_cost:
                            improvement -= 1  # objective function is not improving at this temperature

                        """Test the threshold temperature"""
                        if _cost < t:
                            # accept the threshold
                            s_old = s_new.copy()
                            s_old_cost = get_total_dist(s_old)

                            """Update best solution"""
                            if s_old_cost < s_best_cost:  # update the best route
                                update_counter += 1
                                self.best_route = s_old.copy()  # update best state of the traveling salesman
                                s_best_cost = get_total_dist(self.best_route)

                        loops_counter += 1
        print("Loops:", loops_counter, "Updates:", update_counter)
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
    print("Starting route cost:", get_total_dist(city_list))

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
    sata = SimulatedAnnealingTA(city_list, CoolingSchedule.exponential, benchmark_seconds)
    new_city_list = sata.simulate(2000, 20, True)  # rounds, threshold length, use benchmark instead of rounds
    print("\tBenchmark of", sata.get_bench_time_seconds(), "seconds went over: ", sata.get_time_spent_ms(), "ms")
    print("Final route cost:", (get_total_dist(new_city_list)))
    print(f'\t{get_total_dist(city_list)} -> {get_total_dist(new_city_list)}')
    print("\t\tcost improvement difference", (get_total_dist(city_list) - get_total_dist(new_city_list)))

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
