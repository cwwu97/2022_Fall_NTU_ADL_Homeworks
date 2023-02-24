import json
import argparse

# with open(os.path.join(args.context_file), 'r', encoding='utf-8') as file:
#     context=json.load(file)

def data_to_swag_format(examples, context):
    examples['sent1'] = examples['question']
    examples['sent2'] = ['']*len(examples['sent1'])
    examples['ending0'] = list(map(lambda x: context[x[0]], examples['paragraphs']))
    examples['ending1'] = list(map(lambda x: context[x[1]], examples['paragraphs']))
    examples['ending2'] = list(map(lambda x: context[x[2]], examples['paragraphs']))
    examples['ending3'] = list(map(lambda x: context[x[3]], examples['paragraphs']))

    if 'relevant' in examples.keys():
        examples['label'] = list(map(lambda x, y: x.index(y), examples['paragraphs'], examples['relevant']))

    return examples

def data_to_squad_format(examples, context):
    examples['context'] = list(map(lambda x: context[x], examples['relevant']))
    if 'answer' in examples.keys():
        examples['answers'] = list(map(lambda x: {'answer_start':[x['start']], 'text':[x['text']]}, examples['answer']))

    return examples
