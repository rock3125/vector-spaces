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

if not os.path.exists('related.txt'):
    with open('related.txt', 'wt') as writer:
        counter = 0
        for (word1, v1) in words:
            print("{:05d} {}".format(counter, word1))
            diff_list = []
            vp = numpy.dot(v1, v1)
            for (word2, v2) in words:
                if word1 != word2:
                    vp = numpy.dot(v1, v2)
                    diff_list.append((word2, vp))
            diff_list.sort(key=lambda x: x[1])
            diff_list.reverse()
            # take the top 16
            top_x_list = [word1]
            for (item, score) in diff_list:
                top_x_list.append("{},{:2.4}".format(item, score))
                if len(top_x_list) >= top_x:
                    break
            writer.write(','.join(top_x_list) + '\n')
            counter += 1
