from nvidia.dali.pipeline import Pipeline
from nvidia.dali.plugin import pytorch
import nvidia.dali.ops as ops
import nvidia.dali.types as types

import torch
import math
import numpy as np


class VideoReaderPipeline(Pipeline):
    def __init__(self, batch_size, sequence_length, stride, crop_size, test, num_threads, file_list, seed, step, device_id, shard_id, num_shards, shuffle, mean, std, pad_last_batch):
        super(VideoReaderPipeline, self).__init__(batch_size, num_threads, device_id, seed=seed)
        self.reader = ops.VideoReader(device="gpu", file_list=file_list, sequence_length=sequence_length, normalized=True,
                                    random_shuffle=shuffle, image_type=types.RGB, dtype=types.FLOAT, initial_fill=32, enable_frame_num=True, stride=stride, step=step,
                                    shard_id=shard_id, num_shards=num_shards, stick_to_shard=False, pad_last_batch=pad_last_batch)
#        self.crop = ops.Crop(device="gpu", crop=crop_size, output_dtype=types.FLOAT)
#        self.transpose = ops.Transpose(device="gpu", perm=[3, 0, 1, 2])
        if test:
            self.crop_pos = ops.Constant(fdata=0.5)
            self.is_flip = ops.Constant(idata=0)
        else:
            self.crop_pos = ops.Uniform(range=(0.0, 1.0))
            self.is_flip = ops.CoinFlip(probability=0.5)
        self.cropmirrornorm = ops.CropMirrorNormalize(device="gpu", crop=crop_size, output_dtype=types.FLOAT, mean = mean, std = std, output_layout = "CFHW")

    def define_graph(self):
        input = self.reader(name="Reader")
        crop_pos_x = self.crop_pos()
        crop_pos_y = self.crop_pos()
        is_flipped = self.is_flip()
        output = self.cropmirrornorm(input[0], crop_pos_x=crop_pos_x, crop_pos_y=crop_pos_y, mirror=is_flipped)
#        cropped = self.crop(input[0], crop_pos_x=crop_pos_x, crop_pos_y=crop_pos_y)
#        flipped = self.flip(cropped, horizontal=is_flipped)
#        output = self.transpose(flipped)
        # Change what you want from the dataloader.
        # input[1]: label, input[2]: starting frame number indexed from zero
        return output, input[1], input[2], crop_pos_x, crop_pos_y, is_flipped

class DALILoader():
    def __init__(self, batch_size, file_list, uid2label, sequence_length, stride, crop_size, seed, test = False, step=-1, num_threads=1, device_id=0, shard_id=0, num_shards=1, shuffle=False, mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225], fill_last_batch=False, pad_last_batch=True):
        self.pipeline = VideoReaderPipeline(batch_size=batch_size,
                                            sequence_length=sequence_length,
                                            stride=stride,
                                            num_threads=num_threads,
                                            file_list=file_list,
                                            crop_size=crop_size,
                                            test=test,
                                            seed=seed,
                                            step=step,
                                            device_id=device_id,
                                            shard_id=shard_id,
                                            num_shards=num_shards,
                                            shuffle=shuffle,
                                            mean=mean,
                                            std=std,
                                            # It's always good to pad the last batch, otherwise some data from next epoch are dropped.
                                            # Instead, pad_last_batch changes the size of the shard (epoch)
                                            pad_last_batch=True)
        self.pipeline.build()

        self.uid2label = uid2label
        self.batch_size = batch_size
        self.epoch_size = self.pipeline.epoch_size("Reader")

        if pad_last_batch:
            self.shard_size = math.ceil(self.epoch_size / num_shards)
        else:
            self.shard_size = int(self.epoch_size * (shard_id + 1) / num_shards) - int(self.epoch_size * shard_id / num_shards)

        self.num_iters = self.shard_size // batch_size + int(self.shard_size % batch_size > 0)
        self.dali_iterator = pytorch.DALIGenericIterator(self.pipeline,
                                                         ["data", "uid", "frame_num", "crop_pos_x", "crop_pos_y", "is_flipped"],
                                                         size=self.shard_size,
                                                         auto_reset=True,
                                                         fill_last_batch=fill_last_batch,
                                                         # It's always good to pad the last batch, otherwise some data from next epoch are dropped.
                                                         # Instead, pad_last_batch changes the size of the shard (epoch)
                                                         last_batch_padded=True)
    def __len__(self):
        return int(self.shard_size)
    def __iter__(self):
        #print(self.dali_iterator.__iter__())
        return self
    def __next__(self):
        batch = self.dali_iterator.__next__()[0]
        postprocessed = {}
        postprocessed['data'] = batch['data']
        postprocessed['label'] = torch.from_numpy(np.fromiter((self.uid2label[int(uid)] for uid in batch['uid']), int)).long().to(batch['uid'].device)
        # DALI uses the same buffer so you can't change the shape directly. You must copy them.
        postprocessed['uid'] = batch['uid'].view(-1)
        postprocessed['frame_num'] = batch['frame_num'].view(-1)
        postprocessed['crop_pos_x'] = batch['crop_pos_x'].view(-1)
        postprocessed['crop_pos_y'] = batch['crop_pos_y'].view(-1)
        postprocessed['is_flipped'] = batch['is_flipped'].view(-1)
        return postprocessed 

    # Reset the iterator. Needs to be done at the end of epoch when __next__ is directly called instead of doing iteration.
    def reset(self):
        self.dali_iterator.reset()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_list', type=str, default='file_list.txt',
                        help='DALI file_list for VideoReader')
    parser.add_argument('--frames', type=int, default = 16,
                        help='num frames in input sequence')
    parser.add_argument('--stride', type=int, default = 2,
                        help='temporal stride')
    parser.add_argument('--crop_size', type=int, nargs=2, default=[224, 224],
                        help='[height, width] for input crop')
    parser.add_argument('--batchsize', type=int, default=2,
                        help='per rank batch size')

    args = parser.parse_args()

    from video_datasets_api.epic_kitchens.read_annotations import get_verb_uid2label_dict
    uid2label = get_verb_uid2label_dict('/disk/scratch1/s1884147/datasets/EPIC_KITCHENS_2018/annotations/EPIC_train_action_labels.pkl')
    loader = DALILoader(args.batchsize,
                args.file_list,
                uid2label,
                args.frames,
                args.stride,
                args.crop_size,
                shard_id=3,
                num_shards=8,
                seed=12,
                pad_last_batch=False
                )
    num_samples = len(loader)
    print("Total number of shard samples: %d" % num_samples)
    print("Total number of epoch samples: %d" % loader.epoch_size)

    """
    batch = next(loader)

    print('input shape: %s' % (batch['data'].shape,))
    print('video uids: %s' % batch['uid'])
    print('labels: %s' % batch['label'])
    print('frame nums: %s' % batch['frame_num'])
    print('x crop pos: %s' % batch['crop_pos_x'])
    print('y crop pos: %s' % batch['crop_pos_y'])
    print('is flipped: %s' % batch['is_flipped'])
    """

    sample_seen = 0
    print()
    print(1)
    print()
    for batch in loader:
        sample_seen += batch['data'].shape[0]
        print("%d / %d" % (sample_seen, num_samples))
        #print(batch["uid"])
        #print(batch["frame_num"])

    print()
    print(2)
    print()
    sample_seen = 0
    for batch in loader:
        sample_seen += batch['data'].shape[0]
        print("%d / %d" % (sample_seen, num_samples))
        #print(batch["uid"])
        #print(batch["frame_num"])

    print()
    print(3)
    print()
    sample_seen = 0
    for batch in loader:
        sample_seen += batch['data'].shape[0]
        print("%d / %d" % (sample_seen, num_samples))
        #print(batch["uid"])
        #print(batch["frame_num"])

    print()
    print(4)
    print()
    sample_seen = 0
    for batch in loader:
        sample_seen += batch['data'].shape[0]
        print("%d / %d" % (sample_seen, num_samples))
        #print(batch["uid"])
        #print(batch["frame_num"])
