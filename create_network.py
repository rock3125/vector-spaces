from utils import load_words, word_distance, key, plot
import numpy

# create a neural network like structure
# for vector distances

numpy.random.seed(19680801)

# the vectors [(str, [float])]
words = load_words()

# size of the neural network targets x num_dims
num_dims = 8
show_plot = True

# the words to pick for testing
targets = ['worthy', '-worthy', 'and breathe', 'breathe', 'car', 'truck', 'respire', 'perspire', 'train']

# collect the distances we need for the simulation
distance = dict()
for i in range(0, len(targets)):
    word1 = targets[i]
    for j in range(i + 1, len(targets)):
        word2 = targets[j]
        # use words to lookup word1 and word2 and get their distance
        # in higher dimensional space (from the word vectors)
        dist = word_distance(word1, word2, words)
        # store this at the indices of each word
        distance[key(i, j)] = dist


# initialize a network for the targets
network = []
for i in range(0, len(targets)):
    network.append(numpy.random.uniform(low=-1.0, high=1.0, size=num_dims))

# keep the first concept anchored at the origin point
for i in range(0, num_dims):
    network[0][i] = 0.0

if show_plot:
    plot(network, True)

# learning rate - how big a step to take each time
lr = 0.0001
# when it is ok to give up - stop moving
stop_distance = 0.01
# the number of changes each round, if 0 => stable
num_changes = 1
# iteration counter
counter = 0
# each round, the biggest distance between two points
biggest_dist = 0.0
while num_changes > 0:

    if counter % 100 == 0:
        print("iter {}, changes {}, delta {}".format(counter, num_changes, biggest_dist))
        if show_plot:
            plot(network, False)

    biggest_dist = 0.0
    counter += 1
    num_changes = 0
    for i in range(0, len(targets)):
        for j in range(1, len(targets)):
            if i != j:
                # this is what they should be at
                required_dist = distance[key(i, j)]
                # this is what they are at now
                actual_distance = numpy.linalg.norm(network[i] - network[j])
                # the distance between the two
                delta = abs(actual_distance - required_dist)
                # is it enough to have to update?
                if delta > stop_distance:
                    if delta > biggest_dist:
                        biggest_dist = delta
                    # move j towards i, learning rate step size
                    d_v = network[i] - network[j]
                    # normalize
                    d_v = d_v / numpy.linalg.norm(d_v)
                    # learning rate steps only
                    adjustment_vector = d_v * lr
                    # determine the direction to move in
                    if actual_distance > required_dist:
                        network[j] = network[j] + adjustment_vector
                    else:
                        network[j] = network[j] - adjustment_vector
                    # adjusted - so count it
                    num_changes += 1

