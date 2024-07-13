from transformers import GPT2TokenizerFast, GPT2LMHeadModel
import os
import pickle
import numpy

#
# download the GPT2 vectors
# and create a vectors.txt and a vectors-serialized.bin file
# containing the GPT2 word vectors: 84,575 x 768
#
if not os.path.exists('vectors.txt'):

    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')

    with open('vectors.txt', 'wt') as writer:
        with open('words.txt', 'rt') as reader:
            for word in reader:
                word = word.strip()
                text_index = tokenizer.encode(word)
                if len(text_index) == 1:
                    vector = model.transformer.wte.weight[text_index,:]
                    if len(vector[0]) == 768:
                        v0 = vector[0]
                        vector_list = []
                        for i in range(0, len(v0)):
                            vector = v0[i].detach().numpy()
                            vector_list.append(vector.item())
                        writer.write("{} {}\n".format(word, str(vector_list)))

                elif len(text_index) > 1:
                    vector_list = [0] * 768
                    counter = 0
                    for index in text_index:
                        vector = model.transformer.wte.weight[[index], :]
                        if len(vector) == 1 and len(vector[0]) == 768:
                            counter += 1
                            v0 = vector[0]
                            for i in range(0, len(v0)):
                                vector = v0[i].detach().numpy()
                                vector_list[i] += vector.item()

                    if counter > 1:
                        for i in range(0, 768):
                            vector_list[i] /= counter

                        writer.write("{} {}\n".format(word, str(vector_list)))


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
