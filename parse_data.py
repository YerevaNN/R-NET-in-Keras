"""
Usage: python parse_data.py dataset_file outfile
"""
# -*- coding: utf-8 -*-
import json
import sys

file_path = sys.argv[1]
outfile = 'tmp.json'
if (len(sys.argv) > 2):
    outfile = sys.argv[2]

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

cqas = [] # ContextQuestionAnswer

for topic_id in range(len(data)):
    paragraphs = data[topic_id]['paragraphs']
    for paragraph in paragraphs:
        context = paragraph['context']
        for qa in paragraph['qas']:
            # assert len(qa['answers']) == 1 # for trainset, dev set has a few answers
            
            question = qa['question']
            _id = qa['id']
            answer = qa['answers'][0]['text']
            answer_start = qa['answers'][0]['answer_start'] 
            
            cqas.append({"context": context, "question": question, "answer": answer,
                'answer_start': answer_start, 'id': _id, 'topic': topics[topic_id]
            })
            
print "Dataset contains %d (context, question, answer) triples" % len(cqas)

cnt = [0 for i in range(100)]
max_cnt = 0
for cqa in cqas:
    answer = cqa['answer']
    words = len(answer.split(' '))
    cnt[words] += 1
    max_cnt = max(max_cnt, words)

print "answer length distribution:"
for i in range(max_cnt + 1):
    print "%d: %d" % (i, cnt[i])


print "answer length coverage distribution:"
s = 0    
for i in range(max_cnt + 1):
    s += cnt[i]
    print "%d: %f" % (i, float(s) / len(cqas))

print "Saving dataset to outfile..."
with open(outfile, 'w') as out:
    json.dump(cqas, out)