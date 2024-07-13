import pickle
import os
import numpy
import matplotlib.pyplot as plt

# a figure to optionally show a plot in
fig = plt.figure()


# list of (word: string, vector_list: [float]) tuples
# this is the higher dimensional space from a vector set
def load_words():
    words = []
    if not os.path.exists('vectors-serialized.bin'):
        with open('vectors.txt', 'rt') as reader:
            for line in reader.readlines():
                line = line.strip()
                parts = line.split('[')
                if len(parts) == 2:
                    word = parts[0].strip()
                    vector_str = parts[1].strip()[0:-1]
                    vector_list = []
                    for v in vector_str.split(','):
                        v = float(v.strip())
                        vector_list.append(v)
                    vector_list = vector_list / numpy.linalg.norm(vector_list)
                    words.append((word, vector_list))

        with open('vectors-serialized.bin', 'wb') as writer:
            pickle.dump(words, writer)
    else:
        with open('vectors-serialized.bin', 'rb') as reader:
            words = pickle.load(reader)

    return words


# get the distance between two concepts (slow version)
# the smaller the number, the more related the concepts
def word_distance(word1, word2, words):
    v1 = []
    v2 = []
    for w in words:
        if w[0] == word1:
            v1 = w[1]
            break
    for w in words:
        if w[0] == word2:
            v2 = w[1]
            break

    if len(v1) == 0 or len(v2) == 0:
        raise ValueError("{} or {} not found".format(word1, word2))

    return 1.0 - numpy.dot(v1, v2)


# arrange index 1 and index 2 < and return them as a string index for a dict()
def key(i1, i2):
    if i1 < i2:
        return "{}:{}".format(str(i1), str(i2))
    else:
        return "{}:{}".format(str(i2), str(i1))


# plot the entire network, creating a new plot
def plot(network, force_show):
    if force_show:
        fig.show()
    fig.clear()
    ax = fig.add_subplot(projection='3d')
    # Make legend, set axes limits and labels
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(elev=20., azim=-35, roll=0)

    for i in range(0, len(network)):
        ax.scatter(network[i][0], network[i][1], network[i][2], marker='o')

    fig.canvas.draw()
    fig.canvas.flush_events()
