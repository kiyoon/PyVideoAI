"""
Single positive training label (random choice), multiple label for test labels.
"""
import argparse
from pathlib import Path
import random

from video_datasets_api.wray_multiverb.beoid import read_all_annotations_thresholded


parser = argparse.ArgumentParser(
    "Generate splits of BEOID dataset (Wray)"
)
parser.add_argument(
    "out_folder", type=Path, help="Directory to store split files."
)
parser.add_argument(
    "out_folder_flow", type=Path, help="Directory to store split files (flow)."
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
    segments_info = read_all_annotations_thresholded(str(args.wray_annotations_root_dir), str(args.BEOID_annotations_root_dir))

    args.out_folder.mkdir(exist_ok=True, parents=True)
    args.out_folder_flow.mkdir(exist_ok=True, parents=True)

    num_train_samples = round(len(segments_info) * args.train_ratio / 100)
    for split in range(args.num_splits):
        with open(args.out_folder / f'train{split}.csv', 'w') as train_split:
            with open(args.out_folder / f'val{split}.csv', 'w') as val_split:
                with open(args.out_folder_flow / f'train{split}.csv', 'w') as train_flow_split:
                    with open(args.out_folder_flow / f'val{split}.csv', 'w') as val_flow_split:
                        train_split.write('0\n')
                        val_split.write('42\n')
                        train_flow_split.write('0\n')
                        val_flow_split.write('42\n')
                        random.shuffle(segments_info)
                        for segment in segments_info[:num_train_samples]:
                            label = random.choice(segment.wray_multiverb_idx)
                            write_str = f'{segment.clip_id_str} {segment.clip_id} {label} 0 {segment.end_frame - segment.start_frame}\n'
                            train_split.write(write_str)
                            write_str = f'{segment.clip_id_str} {segment.clip_id} {label} 0 {segment.end_frame - segment.start_frame - 1}\n'
                            train_flow_split.write(write_str)
                        for segment in segments_info[num_train_samples:]:
                            write_str = f'{segment.clip_id_str} {segment.clip_id} {",".join(map(str, segment.wray_multiverb_idx))} 0 {segment.end_frame - segment.start_frame}\n'
                            val_split.write(write_str)
                            write_str = f'{segment.clip_id_str} {segment.clip_id} {",".join(map(str, segment.wray_multiverb_idx))} 0 {segment.end_frame - segment.start_frame - 1}\n'
                            val_flow_split.write(write_str)


if __name__ == "__main__":
    main(parser.parse_args())
