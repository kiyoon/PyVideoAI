import sent2vec # https://github.com/epfml/sent2vec
import gensim.downloader
import dataset_configs
import sklearn.metrics
import numpy as np
import pandas as pd
import os 

DATASET = 'epic55_verb'
#model_type = 'sent2vec'
model_type = 'word2vec'


dataset_cfg = dataset_configs.load_cfg(DATASET)

if model_type == 'sent2vec':
    #PRETRAINED_PATH = '/media/kiyoon/Elements/pretrained/wiki_bigrams.bin'
    PRETRAINED_PATH = '/media/kiyoon/Elements/pretrained/torontobooks_bigrams.bin'
    pretrained_name = os.path.splitext(os.path.basename(PRETRAINED_PATH))[0]

    model = sent2vec.Sent2vecModel()
    model.load_model(PRETRAINED_PATH)

    if DATASET.startswith('epic'):
        class_keys = [class_key.replace('-', ' ') for class_key in dataset_cfg.class_keys]
    else:
        class_keys = list(dataset_cfg.class_keys)

    embeddings = model.embed_sentences(class_keys)

elif model_type == 'word2vec':
    pretrained_name = 'glove-twitter-200'
    #pretrained_name = 'glove-wiki-gigaword-300'
    #pretrained_name = 'word2vec-google-news-300'

    model = gensim.downloader.load(pretrained_name)

    if DATASET.startswith('epic'):
        # only use first word
        class_keys = [class_key[:class_key.find('-')] if class_key.find('-') > 0 else class_key for class_key in dataset_cfg.class_keys]
    else:
        class_keys = list(dataset_cfg.class_keys)

    filtered_class_keys = []
    for class_key in class_keys:
        if class_key in model.index_to_key:
            filtered_class_keys.append(class_key)

    class_keys = filtered_class_keys

    embeddings = model[class_keys] 

else:
    raise ValueError(f'model_type is wrong.')



# write to csv
embedding_pdist = sklearn.metrics.pairwise_distances(embeddings)  # shape (C, C)
embedding_max = embedding_pdist.max()
argsort = np.argsort(embedding_pdist, axis=1)
with open(f'{model_type}-{DATASET}-{pretrained_name}.csv', 'w') as f:
    f.write(','.join(class_keys) + '\n')
    for rank in range(embedding_pdist.shape[0]):
        for class_id, class_embedding in enumerate(embedding_pdist):
            percent = (1 - class_embedding[argsort[class_id, rank]] / embedding_max) * 100
            class_key = class_keys[argsort[class_id, rank]]
            f.write(f'{class_key} {percent:.0f},')

        f.write('\n')


# Confusion matrix
embedding_pdist = 1 - (embedding_pdist / embedding_max)
df_cm = pd.DataFrame(embedding_pdist, class_keys, class_keys)
fig = dataset_cfg.plot_confusion_matrix(df_cm, vmin=0, vmax=1)
fig.savefig(f'{model_type}-{DATASET}-{pretrained_name}.pdf')
