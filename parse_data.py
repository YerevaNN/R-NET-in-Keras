# from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import json
import argparse
import random

if __name__ == '__main__':
    random.seed(42)

    parser = argparse.ArgumentParser()
    parser.add_argument('data', type=str, help='Path to the dataset file')
    parser.add_argument('--outfile', default='data/tmp.json', type=str,
                        help='Desired path to output json')
    parser.add_argument('--train_ratio', default=1., type=float,
                        help='ratio for train/val split')
    args = parser.parse_args()

    with open(args.data, 'r') as f:
        data = json.load(f)

    data = data['data']

    # Lists containing ContextQuestionAnswerS
    train_cqas = []
    valid_cqas = []

    for topic in data:
        cqas = [{'context':      paragraph['context'],
                'id':           qa['id'],
                'question':     qa['question'],
                'answer':       qa['answers'][0]['text'],
                'answer_start': qa['answers'][0]['answer_start'],
                'answer_end':   qa['answers'][0]['answer_start'] + \
                                len(qa['answers'][0]['text']) - 1,
                'topic':        topic['title'] }
                for paragraph in topic['paragraphs']
                for qa in paragraph['qas']]

        if random.random() < args.train_ratio:
            train_cqas += cqas
        else:
            valid_cqas += cqas

    if args.train_ratio == 1.:
        print('Writing dataset to file {}...'.format(args.outfile), end='')
        with open(args.outfile, 'w') as fd:
            json.dump(train_cqas, fd)
        print('Done!')
    else:
        print('Train/Val ratio is {}'.format(len(train_cqas) / len(valid_cqas)))
        train_file = 'train_{}'.format(args.outfile)
        valid_file = 'valid_{}'.format(args.outfile)
        print('Writing datasets to files {}, {}...'.format(train_file,
                                                           valid_file), end='')
        with open(train_file, 'w') as fd:
            json.dump(train_cqas, fd)
        with open(valid_file, 'w') as fd:
            json.dump(valid_cqas, fd)
        print('Done!')
