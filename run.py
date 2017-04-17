import pandas as pd
import numpy as np
import sys
import re
import HiddenMarkovModel as HMM
import os
import traceback

def get_vocab(vocab_file):
    with open(vocab_file) as f:
        lines = f.readlines()

        words = []

        for l in lines:
            words.append(l.split(' ')[1].strip())

    return words

def get_transition_probs(transition_probs_file, offset_idx = True):
    df = pd.read_csv(transition_probs_file, delimiter=' ', header=None, names=['i', 'j', 'prob'])

    num_words = df.shape[0]

    transition_probs = np.zeros((num_words, num_words))

    for idx, row in df.iterrows():   
        i = int(row['i'])
        j = int(row['j'])        
        if offset_idx:
            i -= 1
            j -= 1
        transition_probs[i , j] = 10 ** row['prob']

    return transition_probs

def correct_sentence(HMM, sentence):

    observations = [x for x in re.split('(\W)', sentence.lower()) if x != " " and x != ""]
    
    results = None

    # Correct the sentence with increasing tolerance for errors.
    # This is done for computational efficiency.

    rate = 0.1 # the maximum error rate that we will accept.

    while True:
        try: 
            print("correcting with error rate " + str(rate), end='\r', flush=True)
            results = HMM.viterbi(observations, rate)
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
            traceback.print_tb(exc_tb, limit=10, file=sys.stdout)
            pass

        if results != None:
            return results
        rate += 0.1
        if rate >= 3.0:
            return ""

# settings

"""
start_state = '<s>'
end_state = '</s>'
vocab_file = 'data/vocab.txt'
transition_probs_file = 'data/bigram_counts.txt'
offset_idx = True
"""

start_state = '<S>'
end_state = None
vocab_file = 'data/google_n-gram/vocabulary.txt'
transition_probs_file = 'data/google_n-gram/bigram_count.txt'
offset_idx = False


sentences = [
    "I are",
    'she haf heard them',
    'he said nit word by.'   
]

states = get_vocab(vocab_file)
transition_probs = get_transition_probs(transition_probs_file, offset_idx)

HMM = HMM.HiddenMarkovModel( states, transition_probs, start_state, end_state)

for sentence in sentences:
    result = correct_sentence(HMM, sentence)
    print("\ninput:  " + sentence)
    print("output: " + result)
