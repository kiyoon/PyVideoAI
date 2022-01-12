_exec_relative_('../sparsesample_onehot_RGB_crop224_8frame_largejit_plateau.py')
_exec_relative_('../tsm_resnet50_nopartialbn_base.py')


import torch
from torch import Tensor

import sent2vec
model = sent2vec.Sent2vecModel()
model.load_model('/media/kiyoon/Elements/pretrained/wiki_bigrams.bin')
#import dataset_configs
#dataset_cfg = dataset_configs.load_cfg('epic55_verb')
class_keys = [class_key.replace('-', ' ') for class_key in dataset_cfg.class_keys]
sent2vec_embeddings = model.embed_sentences(class_keys)

# write to csv
embedding_pdist = sklearn.metrics.pairwise_distances(sent2vec_embeddings)  # shape (C, C)
embedding_max = embedding_pdist.max()
import numpy as np
argsort = np.argsort(embedding_pdist, axis=1)
with open('sent2vec_epic.csv', 'w') as f:
    f.write(','.join(class_keys) + '\n')
    for rank in range(embedding_pdist.shape[0]):
        for class_id, class_embedding in enumerate(embedding_pdist):
            percent = (1 - class_embedding[argsort[class_id, rank]] / embedding_max) * 100
            class_key = class_keys[argsort[class_id, rank]]
            f.write(f'{class_key} {percent:.0f},')

        f.write('\n')




import sklearn.metrics
class EmbeddingDistanceLoss(torch.nn.Module):
    def __init__(self, embeddings: np.array) -> None:
        """
        params:
            embeddings: np.array of shape (num_classes, embedding_dim)
        """
        super().__init__()

        self.num_classes = embeddings.shape[0]
        
        self.embedding_pdist = sklearn.metrics.pairwise_distances(embeddings)  # shape (C, C)

    def forward(self, pred_probs: Tensor, target_probs: Tensor) -> Tensor:
        """
        Inputs:
            pred_probs: predictions of shape (N, C).
            target_probs: targets of shape (N,).

        Outputs:
            Loss: a scalar tensor, normalised by N.
        """
        if not (pred_probs.shape[0] == target_probs.shape[0]):
            raise ValueError(f'pred_probs and target_probs has different batch sizes: {pred_probs.shape[0]} and {target_probs.shape[0]}')
        assert pred_probs.dim() == 2, f'pred_probs has to be of size (N, C) but got {pred_probs.dim()} dimensions'
        assert target_probs.dim() == 1, f'target_probs has to be of size (N,) but got {target_probs.dim()} dimensions'
        assert pred_probs.shape[1] == self.num_classes, f'Number of classes is different from embeddings ({self.num_classes}) to predictions ({pred_probs.shape[1]}).'

        distances = self.embedding_pdist[target_probs]  # shape (N, C)
        weighted_distances = distances * pred_probs
        loss = torch.sum(weighted_distances) / pred_probs.shape[0]

        return loss



def get_criterion(split):
    return EmbeddingDistanceLoss(sent2vec_embeddings)


