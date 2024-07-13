from transformers import GPT2TokenizerFast, GPT2LMHeadModel
import os
import pickle
import numpy

#
# download the GPT2 vectors
# and create a vectors.txt and a vectors-serialized.bin file
# containing the GPT2 word vectors: 84,575 x 768
#
if not os.path.exists('vectors-serialized.bin'):

    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    vocab = tokenizer.get_vocab()
    embeddings = model.transformer.wte.weight

    words = []
    with open('vectors-serialized.bin', 'wb') as writer:
        for token, index in list(vocab.items()):
            vector = embeddings[index].tolist()
            if len(vector) == 768:
                vector = vector / numpy.linalg.norm(vector)
                words.append((token, vector))

        pickle.dump(words, writer)
