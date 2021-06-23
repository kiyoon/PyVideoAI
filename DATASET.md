# Overall pipeline

The dataset files should be located at `data/`. The paths of the video/images(frames) and train/val splits should be defined in `dataet_configs/[dataset_name].py`. The dataloader is defined in `exp_configs/[dataset_name]/[model_name]-[experiment_name].py`.

# HMDB-51

1. Download HMDB51 and the official three splits [here](https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/#Downloads).
2. Use `submodules/video_datasets_api/hmdb_tools/hmdb_extract_rar.sh` to extract the rar file.
3. Use `submodules/video_datasets_api/hmdb_tools/hmdb_extract_frames.sh` to extract videos into frames. (optional, it depends on the dataloader you use)  
    a. Because of what is defined in `dataset_configs/hmdb.py`, the frames have to be located at `data/hmdb51/frames_q5'
4. Use `tools/datasets/hmdb_splits_to_csv_frame_extracted.sh` to convert the official three splits to the format required for this package.  
    a. Because of what is defined in `dataset_configs/hmdb.py`, the splits have to be located at `data/hmdb51/splits_frames'

Refer to [examples/01-Prepare_HMDB51.ipynb](https://github.com/kiyoon/PyVideoAI-examples/blob/master/01-Prepare_HMDB51.ipynb) for details.

# Something-Something-V1

1. Go to the TwentyBN's download page. Copy the HTML source code and make save it as `TwentyBN.html`
2. Use `submodules/video_datasets_api/something-something-v1_tools/download_sthsth.py --version 1 TwentyBN.html [download_dir]` to download the dataset.
3. Extract using `cat [download_dir]/20bn-something-something-v1-?? | tar zx -C data/something-something-v1/frames`.  
4. Download annotations and save into `data/something-something-v1/annotations`
5. Use `python tools/datasets/generate_somethingv1_splits.py --root data/something-something-v1/frames data/something-something-v1/splits_frames data/something-something-v1/annotations` to generate splits required to use the PyVideoAI dataloaders.

# CATER (Task 1/2, static/cameramotion)

1. Use `submodules/video_datasets_api/cater_tools/download_cater_max2action.sh [download_dir]` to download and extract both max2action and max2action_cameramotion.  
2. Put the dataset in `data/cater` (make symbolic link)
3. Extract into frames using `submodules/video_datasets_api/cater_tools/cater_extract_frames.sh`.
4. Use `python tools/datasets/generate_cater_task1_2_splits.py` to generate training splits.
