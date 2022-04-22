# CSCI 191T
# Spring 2022
# Traveling Salesperson with Simulated Annealing: Hill climb variant
import math

import matplotlib.pyplot as plt  # plot results
import numpy

# same "randoms" will always be generated because of the seed 
# FIXME: change later
rng = numpy.random.default_rng(seed=420)
print("Seed: ", rng)


def prob(curr_eval, next_eval, tmp):
    if next_eval < curr_eval:
        return 1.0
    else:
        if tmp < 1e-64:  # failsafe for low temp
            return 0.0
        return math.exp(-abs(next_eval - curr_eval) / tmp)


class City:
    def __init__(self, _x, _y):
        self.x = _x
        self.y = _y

    # calculate distance between a and b
    @staticmethod
    def single_dist(_a, _b):
        return numpy.sqrt(numpy.abs(_a.x - _b.x) + numpy.abs(_a.y - _b.y))

    # https://stackoverflow.com/questions/509211/understanding-slicing
    @staticmethod
    def get_total_dist(city_list):
        dist = 0
        # create tuples, everything except last item and last item
        for _a, _b in zip(city_list[:-1], city_list[1:]):
            dist += City.single_dist(_a, _b)
        dist += City.single_dist(city_list[0], city_list[-1])
        return dist


if __name__ == '__main__':
    # 100 random city locations
    cities = []
    for i in range(100):
        cities.append(City(rng.random(), rng.random()))

    # for plotting cities and paths
    fig = plt.figure(figsize=(10, 5))
    axis1 = fig.add_subplot(121)
    axis2 = fig.add_subplot(121)

    for a, b in zip(cities[:-1], cities[1:]):
        axis1.plot([a.x, b.x], [a.y, b.y], 'b')
    axis1.plot([cities[0].x, cities[-1].x], [cities[0].y, cities[-1].y], 'b')
    for curr in cities:
        axis1.plot(curr.x, curr.y, 'ro')

# Simulated Annealing
cost0 = City.get_total_dist(cities)
print("Starting distance: ", cost0)
# can change these values
T = 100
k = 0.8
t0 = T
print("Calculating...")
restart = 0
best = cities.copy()
for i in range(1000):  # number of iterations, terminating condition

    # Exponential cooling schedule
    T = t0 * (k ** i)
    for j in range(100):
        # exchange the values and get a new neighbor
        # randomly picks a neighbor, not ideal
        # want them to be close, use Minimum Spanning Tree?
        r1, r2 = numpy.random.randint(0, len(cities), size=2)

        # swap
        temp = cities[r1]
        cities[r1] = cities[r2]
        cities[r2] = temp

        # new cost value
        cost1 = City.get_total_dist(cities)

        if cost1 < cost0:
            # Only select the best move (hill climbing)
            cost0 = cost1
            best = cities.copy()
        else:
            # Maybe random restart
            p = prob(cost0, cost1, T)
            r = rng.random()
            if r < p:
                restart += 1
                cities = best.copy()

# plot results
print("Final Distance: ", cost0)
print("Restarts: ", restart)

for a, b in zip(cities[:-1], cities[1:]):
    axis2.plot([a.x, b.x], [a.y, b.y], 'b')
axis2.plot([cities[0].x, cities[-1].x], [cities[0].y, cities[-1].y], 'b')
for curr in cities:
    axis2.plot(curr.x, curr.y, 'ro')

plt.show()
