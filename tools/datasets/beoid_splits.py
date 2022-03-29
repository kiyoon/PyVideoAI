import argparse
from pathlib import Path
import random
from collections import OrderedDict

from video_datasets_api.wray_multiverb.beoid import read_all_annotations, BEOIDMultiVerb23Label


parser = argparse.ArgumentParser(
    "Generate splits of BEOID dataset (Wray)"
)
parser.add_argument(
    "out_folder", type=Path, help="Directory to store split files."
)
parser.add_argument(
    "wray_annotations_root_dir",
    type=Path,
)
parser.add_argument(
    "BEOID_annotations_root_dir",
    type=Path,
)
parser.add_argument("--train_ratio", default=70)
parser.add_argument("--num_splits", default=5)
parser.add_argument("--seed", default=12)



def main(args):
    assert 1 < args.train_ratio < 100
    random.seed(args.seed)
    segments_info = read_all_annotations(str(args.wray_annotations_root_dir), str(args.BEOID_annotations_root_dir))
    verb_to_segments: dict[int, list[BEOIDMultiVerb23Label]] = {}
    for segment_info in segments_info:
        verb = segment_info.wray_verblabel_idx
        if verb in verb_to_segments.keys():
            verb_to_segments[verb].append(segment_info)
        else:
            verb_to_segments[verb] = [segment_info]

    # sort by key (verb label)
    verb_to_segments: dict[int, list[BEOIDMultiVerb23Label]] = OrderedDict(sorted(verb_to_segments.items()))

    args.out_folder.mkdir(exist_ok=True, parents=True)

    for split in range(args.num_splits):
        with open(args.out_folder / f'train{split}.csv', 'w') as train_split:
            with open(args.out_folder / f'val{split}.csv', 'w') as val_split:
                train_split.write('0\n')
                val_split.write('0\n')
                for verb, segments in verb_to_segments.items():
                    num_train_samples = round(len(segments) * args.train_ratio / 100)
                    random.shuffle(segments)
                    for segment in segments[:num_train_samples]:
                        write_str = f'{segment.clip_id_str} {segment.clip_id} {segment.wray_verblabel_idx} 0 {segment.end_frame - segment.start_frame + 1}\n'
                        train_split.write(write_str)
                    for segment in segments[num_train_samples:]:
                        write_str = f'{segment.clip_id_str} {segment.clip_id} {segment.wray_verblabel_idx} 0 {segment.end_frame - segment.start_frame + 1}\n'
                        val_split.write(write_str)


if __name__ == "__main__":
    main(parser.parse_args())
