"""
Usage: python preprocess_data.py parsed_file.json --output_destination outfile
"""
# -*- coding: utf-8 -*-
import json
import argparse
import gensim
import numpy as np
import random
import cPickle as pickle
from tqdm import tqdm

random.seed(20)

parser = argparse.ArgumentParser()
parser.add_argument('data', help='Data json', type=str)
parser.add_argument('--output_destination', default='data/tmp.pkl', help='Desired path to output pickle', type=str)
args = parser.parse_args()

file_path = args.data
outfile = args.output_destination
if not outfile.endswith('.pkl'):
    outfile += '.pkl'

print "Reading SQuAD data... ",
with open(file_path) as fd:
    samples = json.load(fd)
print "Done!"

print "Reading word2vec data... ",
word_vec_size = 300
w2v_model = gensim.models.KeyedVectors.load_word2vec_format('word2vec_from_glove_300.vec')
vocab = w2v_model.vocab
print "Done!"

def get_word_vector(word):
    if word in vocab:
        return w2v_model[word]
    else:
        return np.zeros(word_vec_size)

print "Initiating CoreNLP service connection... ",
from stanford_corenlp_pywrapper import CoreNLP
proc = CoreNLP(configdict={'annotators': "tokenize,ssplit"}, corenlp_jars=["/home/tigrann/Documents/stanford-corenlp-full-2017-06-09/*"])
print "Done!"

def parse_sample(context, question, answer_start, answer_end, **kwargs):
    context = proc.parse_doc(context)
    tokens = []
    char_offsets = []
    for s in context['sentences']:
        tokens += s['tokens']
        char_offsets += s['char_offsets']
    
    try:
        answer_start = [answer_start >= s and answer_start < e for s, e in char_offsets].index(True)
        answer_end   = [answer_end   >= s and answer_end   < e for s, e in char_offsets].index(True)
    except ValueError:
        return None
    
    context_vecs = [get_word_vector(token) for token in tokens]
    context_vecs = np.vstack(context_vecs).astype(np.float32)

    question = proc.parse_doc(question)
    tokens = []
    for s in question['sentences']:
        tokens += s['tokens']
    question_vecs = [get_word_vector(token) for token in tokens]
    question_vecs = np.vstack(question_vecs).astype(np.float32)
    return [[context_vecs, question_vecs],
            [answer_start, answer_end]]

print "Parsing samples... ",
samples = [parse_sample(**sample) for sample in tqdm(samples)]
samples = [sample for sample in samples if sample is not None]
print "Done!"

# Transpose
data = [[[], []], 
        [[], []]]
for sample in samples:
    data[0][0].append(sample[0][0])
    data[0][1].append(sample[0][1])
    data[1][0].append(sample[1][0])
    data[1][1].append(sample[1][1])

print "Writing data to file '{}'... ".format(outfile)
with open(outfile, 'w') as fd:
    pickle.dump(data, fd, protocol=pickle.HIGHEST_PROTOCOL)
print "Done!"

print "Bye!"
