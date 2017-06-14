"""
Usage: python parse_data.py dataset_file --output_file outfile.json --train_ratio 1.
"""
# -*- coding: utf-8 -*-
import json
import argparse
import random
random.seed(20)

parser = argparse.ArgumentParser()
parser.add_argument('data', help='Path to the dataset file', type=str)
parser.add_argument('--output_destination', default='data/tmp.json',
                    help='Desired path to output json', type=str)
parser.add_argument('--train_ratio', default=1., help='ratio for train/val split', type=float)
args = parser.parse_args()


file_path = args.data
outfile = args.output_destination
train_ratio = args.train_ratio

json_data = open(file_path, 'r').read()
data = json.loads(json_data)

print "Keys of json are:", data.keys()
data = data['data']
print "Dataset is a list of %d topics, each topic contrains some paragraphs" % len(data)
print "Keys of topics are", data[0].keys() 
topics = [data[i]['title'] for i in range(len(data))]
#print "The topics are:", topics

cnt_paragraphs_in_topic = dict([(data[i]['title'], len(data[i]['paragraphs'])) for i in range(len(data))])
print "Keys of paragraphs are:", data[0]['paragraphs'][0].keys()
print "Dataset contains %d paragraphs in total" % sum(cnt_paragraphs_in_topic[x] for x in cnt_paragraphs_in_topic)
print "Each paragraph has some questions and answers associated with it"
print "Keys of qas sections are:", data[0]['paragraphs'][0]['qas'][0].keys()
print "Keys of answers are:", data[0]['paragraphs'][0]['qas'][0]['answers'][0].keys()

train_cqas = [] # ContextQuestionAnswer
val_cqas = []

for topic_id in range(len(data)):
    paragraphs = data[topic_id]['paragraphs']
    if random.random() < train_ratio:
        train = True
    else:
        train = False
    for paragraph in paragraphs:
        context = paragraph['context']
        for qa in paragraph['qas']:
            # assert len(qa['answers']) == 1 # for trainset, dev set has a few answers
            
            question = qa['question']
            _id = qa['id']
            answer = qa['answers'][0]['text']
            answer_start = qa['answers'][0]['answer_start'] 
            answer_end = answer_start + len(answer) - 1 # answer == context[answer_start : answer_end + 1]
            if train:
                train_cqas.append({"context": context, "question": question, "answer": answer,
                    'answer_start': answer_start, 'answer_end': answer_end,
                    'id': _id, 'topic': topics[topic_id]
                })
            else:
                val_cqas.append({"context": context, "question": question, "answer": answer,
                    'answer_start': answer_start, 'answer_end': answer_end,
                    'id': _id, 'topic': topics[topic_id]
                })
            

print "Saving dataset to outfile..."
if train_ratio == 1.:
    with open(outfile, 'w') as fd:
        json.dump(train_cqas,fd)
else:
    print "Train/Val ratio is %f" % (1. * len(train_cqas) / len(val_cqas))
    train_file = 'train_' + outfile
    val_file = 'val_' + outfile
    with open(train_file, 'w') as fd:
        json.dump(train_cqas, fd)
    with open(val_file, 'w') as fd:
        json.dump(val_cqas, fd)