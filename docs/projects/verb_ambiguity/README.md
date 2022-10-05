# An Action Is Worth Multiple Words: Handling Ambiguity in Action Recognition

Kiyoon Kim, Davide Moltisanti, Oisin Mac Aodha, Laura Sevilla-Lara  
In BMVC 2022.

<img src="https://user-images.githubusercontent.com/12980409/193856345-e0287624-4c84-46af-b245-c07ff263c424.png" alt="Verb Ambiguity" width="800"> 

<img src="https://user-images.githubusercontent.com/12980409/193883304-9e7275ee-2b88-4250-b695-c6d2f0d5acc0.png" alt="Method" width="1000"> 

## Dataset downloads (labels only)
- [EPIC-Kitchens-100-SPMV test set labels](https://github.com/kiyoon/verb_ambiguity/releases/download/datasets-v1.0.0/ek100-val-multiple-verbs-halfagree-halfconfident-include_original-20220427.csv)
- [Confusing-HMDB-102 labels](https://github.com/kiyoon/verb_ambiguity/releases/download/datasets-v1.0.0/confusing_hmdb_102_splits.tar.gz)

## Running feature experiments using pre-extracted features
1. Download pre-extracted features.
- [Download EPIC-Kitchens-100 TSM features](https://github.com/kiyoon/verb_ambiguity/releases/download/datasets-v1.0.0/EPIC_KITCHENS_100_TSM_features.tar.gz)
- [Download EPIC-Kitchens-100 TSM feature neighbours (optional)](https://github.com/kiyoon/verb_ambiguity/releases/download/datasets-v1.0.0/EPIC_KITCHENS_100_TSM_neighbour_cache.tar.gz): Using this neighbour cache will reduce the preparation time of the training by skipping neighbour search.
- [Download Confusing-HMDB-102 TSM features](https://github.com/kiyoon/verb_ambiguity/releases/download/datasets-v1.0.0/confusing_hmdb_102_TSM_features.tar.gz)

2. Exract in `PyVideoAI/data/EPIC_KITCHENS_100` or `PyVideoAI/data/hmdb51`.
3. Run the training code. Change the dataset and exp_name variables to select different experiments.
 
```bash
exp_root="$HOME/experiments"  # Experiment results will be saved here.

export CUDA_VISIBLE_DEVICES=0
export VAI_USE_NEIGHBOUR_CACHE=True     # Only for EPIC-Kitchens-100-SPMV. It will bypass neighbour search if the cache is available, otherwise it will run and cache the results.
export VAI_NUM_NEIGHBOURS=15
export VAI_PSEUDOLABEL_THR=0.1

subfolder="k=$VAI_NUM_NEIGHBOURS,thr=$VAI_PSEUDOLABEL_THR"           # Name subfolder as you like.

dataset=epic100_verb_features
#dataset=confusing_hmdb_102_features

exp_name="concat_RGB_flow_assume_negative"
#exp_name="concat_RGB_flow_weak_assume_negative"
#exp_name="concat_RGB_flow_binary_labelsmooth"
#exp_name="concat_RGB_flow_binary_negative_labelsmooth"
#exp_name="concat_RGB_flow_binary_focal"
#exp_name="concat_RGB_flow_entropy_maximise"
#exp_name="concat_RGB_flow_mask_binary_ce"
#exp_name="concat_RGB_flow_pseudo_single_binary_ce"

# Training script
# -S creates a subdirectory in the name of your choice. (optional)
PyVideoAI/tools/run_singlenode.sh train 1 -R $exp_root -D $dataset -c:d verbambig -M ch_beta.featuremodel -E $exp_name -c:e verbambig -S "$subfolder" #--wandb_project kiyoon_kim_verbambig

# Evaluating script
# -l -2 loads the best model (with the highest heldout validation accuracy)
# -p saves the predictions. (optional)
PyVideoAI/tools/run_singlenode.sh eval 1 -R $exp_root -D $dataset -c:d verbambig -M ch_beta.featuremodel -E $exp_name -c:e verbambig -S "$subfolder" -l -2 -p #--wandb
```

## Citing the paper

If you find our work or code useful, please cite:

```BibTeX
@inproceedings{kim2022ambiguity,
  title={An Action Is Worth Multiple Words: Handling Ambiguity in Action Recognition},
  author={Kim, Kiyoon and Moltisanti, Davide and Mac Aodha, Oisin and Sevilla-Lara, Laura},
  booktitle={British Machine Vision Conference (BMVC)},
  year={2022}
}
