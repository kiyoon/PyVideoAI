import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from tqdm import tqdm


def get_random_neighbours(source, target, source_labels, target_labels, kn, n_classes=97, **kwargs):

    source_n = len(source)
    target_n = len(target)

    if id(source) == id(target):
        remove_query = True
    else:
        remove_query = False

    if remove_query:
        random_neighbours = []

        for i in tqdm(range(target_n), desc='Generating random samples'):
            rn = None

            while rn is None or i in rn:
                rn = np.random.choice(source_n, kn, replace=False)

            random_neighbours.append(rn)

        random_neighbours = np.stack(random_neighbours, axis=0)
    else:
        random_neighbours = np.random.choice(source_n, (target_n, kn), replace=True)

    cmf, nc_freq = get_ncf_from_neighbours(random_neighbours, n_classes, source_labels, target_labels)

    return nc_freq, cmf, random_neighbours


def get_neighbours(source, target, source_labels, target_labels, kn, n_classes, metric='euclidean', n_jobs=-1, l2_norm=False):

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

    print(f'Getting neighbours, k={kn}, remove query={remove_query}')

    neigh = NearestNeighbors(n_neighbors=kn, n_jobs=n_jobs)
    neigh.fit(source)  # uses Euclidean distance by default  # TODO try cosine distance
    # from the doc: If X is not provided, neighbors of each indexed point are returned.
    # In this case, the query point is not considered its own neighbor.
    features_neighbours = neigh.kneighbors(X=None if remove_query else target, return_distance=False)

    cmf, nc_freq = get_ncf_from_neighbours(features_neighbours, n_classes, source_labels, target_labels)

    return nc_freq, cmf, features_neighbours


def get_ncf_from_neighbours(features_neighbours, n_classes, source_labels, target_labels):
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

    return cmf, nc_freq
