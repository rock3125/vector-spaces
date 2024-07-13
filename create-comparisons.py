import os
import pickle
import numpy

top_x = 16

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
