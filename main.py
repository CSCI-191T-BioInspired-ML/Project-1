# CSCI 191T
# Spring 2022
# Traveling Salesperson with Simulated Annealing
#  
# Basic Simulated Annealing

import numpy
import matplotlib.pyplot as plt # plot results

# same "randoms" will always be generated because of the seed 
# FIXME: change later
rng = numpy.random.default_rng(seed=420)
print("Seed: ", rng)

class City:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    # calculate distance between a and b
    @staticmethod
    def singleDistance(a, b):
        return numpy.sqrt(numpy.abs(a.x - b.x) + numpy.abs(a.y - b.y))

    # https://stackoverflow.com/questions/509211/understanding-slicing
    @staticmethod
    def getTotalDistance(cities):
        dist = 0
        # create tuples, everything except last item and last item
        for a, b in zip(cities[:-1], cities[1:]):
            dist += City.singleDistance(a, b)
        dist += City.singleDistance(cities[0], cities[-1])
        return dist

if __name__ == '__main__':
    # 20 random city locations
    cities = []
    for i in range (20):
       # cities.append(City(numpy.random.uniform(), numpy.random.uniform())) # apparently the depreciated way?
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
cost0 = City.getTotalDistance(cities)
print("Starting distance: ", cost0)
# can change these values
T = 30  
factor = 0.99
T_init = T

print("Calculating...")
for i in range(1000):   # number of iterations, terminating condition
    # Debug:
    # print(i, "const", cost0)

    # cooling schedule
    T = T * factor
    for j in range(100):
        # exchange the values and get a new neighbor
        # randomly picks a neighbor, not ideal
        # want them to be close, use Minimum Spanning Tree?
        r1, r2 = numpy.random.randint(0, len(cities), size = 2)

        # swap
        temp = cities[r1]
        cities[r1] = cities[r2]
        cities[r2] = temp

        # new cost value
        cost1 = City.getTotalDistance(cities)

        if cost1 < cost0:
            cost0 = cost1
        else:
            x = numpy.random.uniform()
            if x < numpy.exp((cost0 - cost1)/T):
                cost0 = cost1
            else:
                temp = cities[r1]
                cities[r1] = cities[r2]
                cities[r2] = temp
# plot results
print("Final Distance: ", cost0)

for a, b in zip(cities[:-1], cities[1:]):
    axis2.plot([a.x, b.x], [a.y, b.y], 'b')
axis2.plot([cities[0].x, cities[-1].x], [cities[0].y, cities[-1].y], 'b')
for curr in cities:
    axis2.plot(curr.x, curr.y, 'ro')

plt.show()