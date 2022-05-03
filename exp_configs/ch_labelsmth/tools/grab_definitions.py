"""
Grab dictionary and Wikipedia definitions
"""
import requests
import json
import dataset_configs
from exp_configs.ch_labelsmth.tools.sent2vec_utils import get_sentence_embeddings

DATASET = 'epic100_verb'
dataset_cfg = dataset_configs.load_cfg(DATASET)

words = []
word_definitions = []

for class_key in dataset_cfg.class_keys:
    word = class_key.replace('-', ' ')
    print(word)
    word_idx = 0
    ret = requests.get(f'https://api.dictionaryapi.dev/api/v2/entries/en/{word}')
    if ret.status_code == 200:
        word_results = json.loads(ret.text)

        for word_result in word_results:
            for meanings in word_result['meanings']:
                if meanings['partOfSpeech'] == 'verb':
                    for definition in meanings['definitions']:
                        print(definition)
                        words.append(f'{word}{word_idx}')
                        word_definitions.append(f'{definition["definition"]}')
    else:
        print("NO DEFINITIONS FOUND")
        print("Using the word itself for definition.")
        words.append(word)
        word_definitions.append(word)

    print()
