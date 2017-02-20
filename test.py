import pandas as pd
import numpy as np
import sys

def get_vocab():
    fname = 'data/vocab.txt'
    with open(fname) as f:
        lines = f.readlines()

        words = []

        for l in lines:
            words.append(l.split(' ')[1].strip())

    return words

def get_transition_probs():
    df = pd.read_csv('data/bigram_counts.txt', delimiter=' ', header=None, names=['i', 'j', 'prob'])

    num_words = df.shape[0]

    transition_probs = np.zeros((num_words, num_words))

    for idx, row in df.iterrows():
        transition_probs[int(row['i']-1), int(row['j']-1)] = 10 ** row['prob']

    return transition_probs

states = get_vocab()
print((len(states)))
#states = [x for x in states if len(x)<=4]
print(len(states))
transition_probs = get_transition_probs()






start_state_idx = states.index('<s>')
end_state_idx = states.index('</s>')


import HiddenMarkovModel as HMM
def correct_sentence(HMM, sentence):
    return HMM.viterbi(sentence.split(' '))

HMM = HMM.HiddenMarkovModel( states, transition_probs, start_state_idx, end_state_idx )
#sentence = 'she haf heard them'
sentence = 'he said nit word by'
result = correct_sentence(HMM, sentence)
print(result)