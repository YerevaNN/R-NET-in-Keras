# from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
import json
import argparse
import cPickle as pickle

from os import path
from tqdm import tqdm

from utils import CoreNLP_path
from stanford_corenlp_pywrapper import CoreNLP
from gensim.models import KeyedVectors

def CoreNLP_tokenizer():
    proc = CoreNLP(configdict={'annotators': 'tokenize,ssplit'},
                   corenlp_jars=[path.join(CoreNLP_path(), '*')])

    def tokenize_context(context):
        parsed = proc.parse_doc(context)
        tokens = []
        char_offsets = []
        for sentence in parsed['sentences']:
            tokens += sentence['tokens']
            char_offsets += sentence['char_offsets']
        
        return tokens, char_offsets

    return tokenize_context

def word2vec(word2vec_path):
    model = KeyedVectors.load_word2vec_format(word2vec_path)

    def get_word_vector(word):
        try:
            return model[word]
        except KeyError:
            return np.zeros(model.vector_size)

    return get_word_vector

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--word2vec_path', type=str, 
                        default='data/word2vec_from_glove_300.vec',
                        help='Word2Vec vectors file path')
    parser.add_argument('--outfile', type=str, default='data/tmp.pkl',
                        help='Desired path to output pickle')
    parser.add_argument('data', type=str, help='Data json')
    args = parser.parse_args()

    if not args.outfile.endswith('.pkl'):
        args.outfile += '.pkl'

    print('Reading SQuAD data... ', end='')
    with open(args.data) as fd:
        samples = json.load(fd)
    print('Done!')

    print('Initiating CoreNLP service connection... ', end='')
    tokenize = CoreNLP_tokenizer()
    print('Done!')

    print('Reading word2vec data... ', end='')
    word_vector = word2vec(args.word2vec_path)
    print('Done!')

    def parse_sample(context, question, answer_start, answer_end, **kwargs):
        tokens, char_offsets = tokenize(context)
        try:
            answer_start = [answer_start >= s and answer_start < e
                            for s, e in char_offsets].index(True)
            answer_end   = [answer_end   >= s and answer_end   < e
                            for s, e in char_offsets].index(True)
        except ValueError:
            return None

        context_vecs = [word_vector(token) for token in tokens]
        context_vecs = np.vstack(context_vecs).astype(np.float32)

        tokens, char_offsets = tokenize(question)
        question_vecs = [word_vector(token) for token in tokens]
        question_vecs = np.vstack(question_vecs).astype(np.float32)
        return [[context_vecs, question_vecs],
                [answer_start, answer_end]]

    print('Parsing samples... ', end='')
    samples = [parse_sample(**sample) for sample in tqdm(samples)]
    samples = [sample for sample in samples if sample is not None]
    print('Done!')

    # Transpose
    data = [[[], []], 
            [[], []]]
    for sample in samples:
        data[0][0].append(sample[0][0])
        data[0][1].append(sample[0][1])
        data[1][0].append(sample[1][0])
        data[1][1].append(sample[1][1])

    print('Writing to file {}... '.format(args.outfile), end='')
    with open(args.outfile, 'wb') as fd:
        pickle.dump(data, fd, protocol=pickle.HIGHEST_PROTOCOL)
    print('Done!')
