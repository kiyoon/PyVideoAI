import pickle
from ast import literal_eval
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from natsort import natsorted
from video_datasets_api.epic_kitchens_100.read_annotations import epic_narration_id_to_unique_id


def get_participant_video_id_from_vid(vid, vid_to_nid):
    nid_splits = vid_to_nid[vid].split('_')
    participant_id = nid_splits[0]
    video_id = '_'.join(nid_splits[:2])
    return participant_id, video_id


def get_vid_to_narration_id(ek_annotations_path):
    narration_id_to_video_id, narration_ids_sorted = epic_narration_id_to_unique_id(ek_annotations_path)
    vid_to_narration_id = {v: k for k, v in narration_id_to_video_id.items()}

    return vid_to_narration_id, narration_id_to_video_id


def get_verb_maps(ek_annotations_path):
    verb_df_path = Path(ek_annotations_path) / 'EPIC_100_verb_classes.csv'
    df_verbs = pd.read_csv(verb_df_path)
    verb_id_to_str = {}
    verb_str_to_id = {}

    for t in df_verbs.itertuples():
        verb_id_to_str[t.id] = t.key
        verb_str_to_id[t.key] = t.id

    return verb_id_to_str, verb_str_to_id, df_verbs


def get_cm(labels, preds, n_classes, title):
    cm = np.zeros((n_classes, n_classes))

    for y, p in zip(labels, preds):
        y_ = np.argmax(p)
        cm[y][y_] += 1

    for ir, r in enumerate(cm):  # normalise matrix
        s = r.sum()

        if s > 0:
            cm[ir] = r / s

    fig, ax = plt.subplots(dpi=150, facecolor='white')
    ax.imshow(cm)
    ax.set_xticks(range(n_classes))
    ax.set_xticklabels(range(n_classes), fontsize=3, rotation=90)
    ax.set_yticks(range(n_classes))
    ax.set_yticklabels(range(n_classes), fontsize=3)
    ax.set_title(f'Confusion matrix {title}', fontsize=6)

    return cm


def load_predictions(pred_path, feature_ids, feature_labels):
    with open(pred_path, 'rb') as f:
        predictions = pickle.load(f)

    sort_idx_p = np.argsort(predictions['video_ids'])

    for k, v in predictions.items():
        predictions[k] = v[sort_idx_p]

    sort_idx_f = np.argsort(feature_ids)

    assert np.array_equal(feature_ids[sort_idx_f], predictions['video_ids']) and \
           np.array_equal(feature_labels[sort_idx_f], predictions['video_labels'])

    return predictions


def get_params_from_path(res_path):
    res_path = Path(res_path)
    p_str = res_path.parent.name
    splits = p_str.split(';')
    params = {}

    for s in splits:
        ss = s.split('=')
        k = ss[0]
        v = '='.join(ss[1:])

        try:
            v = float(v)
        except ValueError:
            v = str(v)

        params[k] = v

    return params


def collate_results(output_path, metric='top1', value='max', filename='summary.csv', times_100=False,
                    precision=2, apply_style=True, cmap='Greens', ignore_params=(), filter_dict=None,
                    filter_if_not_present=(), res_parts=2, filter_mode='include', sort_by=None,
                    extra_metrics=(), extra_metrics_val=(), rename_p=None):
    # files = glob.glob(os.path.join(output_path, '**', filename), recursive=True)
    files = Path(output_path).rglob(f'**/{filename}')
    rows = []
    output_path_parts = set(Path(output_path).parts)
    assert len(extra_metrics) == len(extra_metrics_val), 'Provide min/max val for each extra metric'
    filter_dict = {} if filter_dict is None else filter_dict

    for res_path in files:
        res_path = Path(res_path)
        run_params = get_params_from_path(res_path)
        df = pd.read_csv(res_path)
        path_diff = set(res_path.parts[:-res_parts]) - output_path_parts
        run_params.update({f'p_{i}': p for i, p in enumerate(sorted(path_diff))})

        if isinstance(filter_if_not_present, dict):
            keep = all([run_params.get(k, None) == v for k, v in filter_if_not_present.items()])

            if not keep:
                continue
        else:
            if filter_if_not_present and not set(filter_if_not_present).intersection(set(run_params.keys())):
                continue

        add_row = True if len(filter_dict) == 0 or filter_mode == 'exclude' else False
        filter_inc_count = 0

        for k, v in run_params.items():
            if k in ignore_params:
                continue

            if filter_dict.get(k, None) == v:
                if filter_mode == 'exclude':
                    add_row = False
                    break
                else:
                    filter_inc_count += 1

        if filter_mode == 'include' and len(filter_dict) > 0:
            add_row = filter_inc_count == len(filter_dict)

        if add_row:
            row = run_params.copy()
            val = df[metric].max() if value == 'max' else df[metric].min()
            row[metric] = val * 100 if times_100 else val

            for em, emv in zip(extra_metrics, extra_metrics_val):
                if em not in df:
                    ev = None
                else:
                    ev = df[em].max() if emv == 'max' else df[em].min()

                row[em] = ev

            row['path'] = res_path
            rows.append(row)

    df = pd.DataFrame(rows)

    if ignore_params:
        df = df.drop(columns=list(ignore_params))

    fixed_params = {}
    varying_params = []

    for c in df.columns:
        if c == metric:
            continue

        u_p = df[c].unique()

        if len(u_p) == 1:
            fixed_params[c] = u_p[0]
        else:
            varying_params.append(c)

    df = df[varying_params + [metric] + list(extra_metrics)]

    if len(varying_params) == 1:
        df = df.set_index(varying_params[0]).sort_index()
    elif len(varying_params) == 2 and len(extra_metrics) == 0:
        index, new_col = varying_params[0], varying_params[1]
        new_rows = []
        indices = df[index].unique()

        for i in indices:
            a = df[df[index] == i][[new_col, metric]]
            d = {'{}={}'.format(new_col, getattr(t[1], new_col)): getattr(t[1], metric) for t in a.iterrows()}
            d[index] = i
            new_rows.append(d)

        df = pd.DataFrame(new_rows)
        df = df.set_index(index)
        df = df.reindex(index=natsorted(df.index))
        df = df.reindex(columns=natsorted(df.columns))

    if sort_by is not None and len(varying_params) != 2:
        df = df.sort_values(by=sort_by, ignore_index=True)

    if rename_p is not None:
        df = df.rename(columns=rename_p)

    if apply_style:
        styled_df = df.style.background_gradient(cmap=cmap, axis=None).set_precision(precision)

        if value == 'max':
            styled_df = styled_df.highlight_max(axis=None, color='darkorange')
        else:
            styled_df = styled_df.highlight_min(axis=None, color='darkorange')
    else:
        styled_df = None

    paths = {t.Index: t.path for t in df.itertuples()}
    df = df.drop(columns='path')

    return df, fixed_params, styled_df, paths
