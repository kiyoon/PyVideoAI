from pathlib import Path

import numpy as np
from scipy import linalg
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from tqdm import tqdm

import .utils
#from dataset import TrainDataset
from .utils import get_participant_video_id_from_vid


def get_neighbours(source, target, source_labels, target_labels, kn, metric='euclidean', n_classes=97, n_jobs=-1,
                   leave_video_out=False, leave_participant_out=False, vid_to_nid=None, source_ids=None,
                   target_ids=None, l2_norm=False):
    if id(source) == id(target):
        remove_query = True
    else:
        remove_query = False

    if l2_norm:
        print('Applying L2 normalisation to features')
        source = normalize(source, norm='l2', axis=1)

        if remove_query:
            target = source
        else:
            target = normalize(target, norm='l2', axis=1)

    print(f'Getting neighbours, k={kn}, remove query={remove_query}, leave video out={leave_video_out}, '
          f'leave participant out={leave_participant_out}')

    if leave_participant_out or leave_video_out:
        assert source_ids is not None and target_ids is not None and vid_to_nid is not None
        # D_{i, j} is the distance between the ith array from X and the jth array from Y.
        print('Calculating pair-wise distances')
        D = pairwise_distances(X=target, Y=source, metric=metric, n_jobs=n_jobs)

        video_idx_map = dict(source=dict(), target=dict())
        participant_idx_map = dict(source=dict(), target=dict())
        max_ = D.max() + 1000

        for set_, vids in zip(('source', 'target'), (source_ids, target_ids)):
            for i, vid in enumerate(vids):
                participant_id, video_id = get_participant_video_id_from_vid(vid, vid_to_nid)

                if video_id in video_idx_map[set_]:
                    video_idx_map[set_][video_id].append(i)
                else:
                    video_idx_map[set_][video_id] = [i]

                if participant_id in participant_idx_map[set_]:
                    participant_idx_map[set_][participant_id].append(i)
                else:
                    participant_idx_map[set_][participant_id] = [i]

        for i, vid in enumerate(target_ids):
            participant_id, video_id = get_participant_video_id_from_vid(vid, vid_to_nid)

            if leave_participant_out:
                to_remove = participant_idx_map['source'].get(participant_id, [])
            else:
                to_remove = video_idx_map['source'].get(video_id, [])

            D[i, to_remove] = max_

        neigh = NearestNeighbors(metric='precomputed', n_neighbors=kn, n_jobs=n_jobs)
        neigh.fit(D)
        features_neighbours = neigh.kneighbors(X=None if remove_query else D, return_distance=False)
    else:
        neigh = NearestNeighbors(n_neighbors=kn, n_jobs=n_jobs)
        neigh.fit(source)  # uses Euclidean distance by default  # TODO try cosine distance

        # from the doc: If X is not provided, neighbors of each indexed point are returned.
        # In this case, the query point is not considered its own neighbor.
        features_neighbours = neigh.kneighbors(X=None if remove_query else target, return_distance=False)

    nc_freq = []

    for fn in features_neighbours:
        cn = [source_labels[x] for x in fn]
        nc, freq = np.unique(cn, return_counts=True)
        fq = np.zeros(n_classes, dtype=int)

        for ncc, fqq in zip(nc, freq):
            fq[ncc] = fqq

        nc_freq.append(fq)

    nc_freq = np.array(nc_freq)
    cmf = np.zeros((n_classes, n_classes))

    for ir, r in enumerate(nc_freq):
        gt_label = target_labels[ir]

        for ic, f in enumerate(r):
            cmf[gt_label, ic] += f

    for ir, r in enumerate(cmf):  # normalise matrix
        s = r.sum()

        if s > 0:
            cmf[ir] = r / s

    return nc_freq, cmf, features_neighbours





def debug():
    root_video_path = Path('/home/davide/data/EPIC-KITCHENS-100-val/val/')
    ek_annotations_path = '/home/davide/Code/UOE/epic-kitchens-100-annotations/'
    vid_to_narration_id, narration_id_to_video_id = utils.get_vid_to_narration_id(ek_annotations_path)
    verb_id_to_str, verb_str_to_id, df_verbs = utils.get_verb_maps(ek_annotations_path)
    kn = 5
    val_features_path = '/home/davide/data/multi-label-ar/epic100_features_tsm_flow/features_epoch_0009_val_oneclip.pkl'
    val_features, val_labels, val_video_ids, n_classes, _ = TrainDataset.load_data(val_features_path, pred_path=None,
                                                                                   n_classes=97, needs_avg=False)
    val_nc_freq, val_cmf, val_neighbours = get_neighbours(val_features, val_features, val_labels, val_labels, kn,
                                                          leave_participant_out=True, vid_to_nid=vid_to_narration_id,
                                                          target_ids=val_video_ids, source_ids=val_video_ids)

    vid = 67881
    idx = np.where(val_video_ids == vid)[0][0]

    for neigh in val_neighbours[idx]:
        print(vid_to_narration_id[val_video_ids[neigh]])


if __name__ == '__main__':
    debug()
