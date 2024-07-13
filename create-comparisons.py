import os
import numpy

from utils import load_words

words = load_words()
if not os.path.exists('distance.txt'):
    with open('distance.txt', 'wt') as writer:
        counter = 0
        diff_list = []
        for i in range(0, len(words)):
            word1 = words[i][0]
            v1 = words[i][1]
            print("{:05d} {}".format(counter, word1))
            for j in range(i + 1, len(words)):
                word2 = words[j][0]
                v2 = words[j][1]
                vp = numpy.dot(v1, v2)
                writer.write("{},{},{}\n".format(word1, word2, vp))
            counter += 1
