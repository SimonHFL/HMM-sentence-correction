import pandas as pd
import numpy as np
import sys
import re
import HiddenMarkovModel as HMM

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

def correct_sentence(HMM, sentence):

    observations = [x for x in re.split('(\W)', sentence) if x != " " and x != ""]
    
    results = None
    rate = 0.1
    while True:
        try: 
            results = HMM.viterbi(observations, rate)
        except:
            pass
        if results != None:
            return results
        rate += 0.1
        if rate >= 3.0:
            return ""

states = get_vocab()
transition_probs = get_transition_probs()
start_state_idx = states.index('<s>')
end_state_idx = states.index('</s>')

HMM = HMM.HiddenMarkovModel( states, transition_probs, start_state_idx, end_state_idx )
sentence = 'she haf heard them'
#sentence = 'he said nit word by.'
result = correct_sentence(HMM, sentence)
print(result)