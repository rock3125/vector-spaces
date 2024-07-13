import matplotlib.pyplot as plt
from utils import load_words, word_distance, key, plot
import numpy

# create a neural network like structure
# for vector distances

numpy.random.seed(19680801)

# the vectors [(str, [float])]
words = load_words()

# size of the neural network
num_dims = 8
show_plot = True

targets = ['worthy', '-worthy', 'and breathe', 'breathe', 'car', 'truck', 'respire', 'perspire', 'train']

distance = dict()
for i in range(0, len(targets)):
    word1 = targets[i]
    for j in range(i + 1, len(targets)):
        word2 = targets[j]
        # the closer, the more related
        dist = word_distance(word1, word2, words)
        distance[key(i, j)] = dist


# initialize a network for the targets
network = []
for i in range(0, len(targets)):
    network.append(numpy.random.uniform(low=-1.0, high=1.0, size=num_dims))

# keep the first concept anchored at the origin point
for i in range(0, num_dims):
    network[0][i] = 0.0

fig = None
if show_plot:
    fig = plt.figure()
    plot(fig, network)
    fig.show()

lr = 0.0001
stop_distance = 0.01
num_changes = 1
counter = 0
biggest_dist = 0.0
while num_changes > 0:

    if counter % 100 == 0:
        print("count {}, num changes {}, delta-max {}".format(counter, num_changes, biggest_dist))
        if fig is not None:
            plot(fig, network)

    biggest_dist = 0.0
    counter += 1
    num_changes = 0
    for i in range(0, len(targets)):
        for j in range(1, len(targets)):
            if i != j:
                required_dist = distance[key(i, j)]
                actual_distance = numpy.linalg.norm(network[i] - network[j])
                delta = abs(actual_distance - required_dist)
                if delta > stop_distance:
                    if delta > biggest_dist:
                        biggest_dist = delta
                    # move j towards i by learning rate step size
                    d_v = network[i] - network[j]
                    d_v = d_v / numpy.linalg.norm(d_v)
                    adjustment_vector = d_v * lr
                    if actual_distance > required_dist:
                        network[j] = network[j] + adjustment_vector
                    else:
                        network[j] = network[j] - adjustment_vector
                    num_changes += 1

