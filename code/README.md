# Explanation of contents

```
all_data_script.py -> script that initiates dataset creation
all_split_script.py -> script that initiates the data split creation process
bae_net/bae_net.py -> contains network arch for BAE-Net
bae_net/gen_data.py -> contains data preprocessing logic for BAE-Net
data_splits -> folder with train  / val / test split information
data_utils.py -> helper file for functions related to data
environment.yml -> required packages 
eval_sem_label_models.py -> helper functions for evaluation of sem label likelihood networks
eval_utils.py -> helper files for function related to evaluation
fastwinding -> computes inside/outside value of points
focal_loss.py -> defining focal loss
grammar.py -> extracting information from PartNet hierarchies, and turning them into full shape grammars
infer_lhss.py -> helper to run inference with already trained lhss models
layers -> subset of GNN layers from https://github.com/graphdeeplearning/benchmarking-gnns
lel_net/ss_loss.py -> self-supervised loss for lel training
lhss/mrf.py -> setup MRF for lhss
lhss/pygco/mrf_solve.py -> solve MRF with alpha-expansion
lhss/feat_code/calc_feat.py -> call out to LHSS Julia code to compute per shape features
lik_mods/sem_label_lik.py -> main entry for semantic label likelihood networks
lik_mods/reg_group_lik.py -> main entry for region group likelihood networks
lik_mods/model_output/ -> output of likelihood model runs
lik_mods/lik_output/ -> logged output of likelihood evaluation
lik_mods/arti_props/ -> proposal perturbations used to train the likelihood networks
make_areas.py -> script that generated area.zip, records area of each region of each shape in partnet
make_arti_data.py -> script that generated arti_props.zip, used to generate artifical proposals used to train semantic label likelihood networks
make_bae_data.py -> cache all BAE-Net shape preprocessing
make_dataset.py -> subs-script that for each shape for a particular cateogry, find all regions (e.g. .obj files) and record their semantic label from PartNet
make_lhss_data.py -> cache all LHSS shape preproccesing
make_splits.py -> sub-script that created the data splits between train / val / test
manifold -> script to turn mesh into a manifold mesh
model_output -> output of model runs
models.py -> defining model architectures
nets -> sub-set of GNN networks architectures from https://github.com/graphdeeplearning/benchmarking-gnns
ngsp_eval.py -> run NGSP inference with trained guide and likelihood networks
pc_enc -> code for point cloud auto-encoding
pc_enc/pc_ae.py -> contains the training code for the per-region point cloud auto-encoder.
pc_enc/pairpc_ae.py -> contains the training code for the paired-region point cloud auto-encoder.
pointnet2 -> helper functions / build operations needed for pointNet++ models
sem_label_data_utils.py -> helper files for training likelihood models (geometry and layout networks)
tmp -> used to store intermediate feature preprocessing results
train_bae_net.py -> BAE-NET entrypoint
train_guide_net.py -> helper functions for guide network training
train_lel_net.py -> LEL entrypoint
train_lhss.py -> LHSS entrypoint
train_lik_models.py -> boiler-plate logic for semantic label likelihood and region group likelihood network training
train_ngsp.py -> script for full NGSP training+evaluation run
train_partnet.py -> PartNet model entrypoint
train_sem_label_models.py -> helper functions for geometry and layout network training
train_utils.py -> helper functions for training
utils.py -> general helper functions
vis_qual_colors.py -> coloring logic used during rendering of qualitative results for the main paper and supplemental pdf
```
