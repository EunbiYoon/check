# data

- blind rollout (1 node)   : `CUDA_VISIBLE_DEVICES=0,1,2 BLIND_VARIANTS=core,aux,all ./data/alpha-beta/blind_rollout.sh`
- blind rollout (4×3 GPU)   : `./data/alpha-beta/blind-rollout/shard.sh <1|2|3|4>` on each node, then `./data/alpha-beta/blind-rollout/shard.sh merge`
- A+β pairs                 : `python data/alpha-beta/noleakage-frontier/build_pairs.py --input data/alpha-beta/blind-rollout/result/a_beta_all.jsonl --output data/paper/a_beta_all.jsonl --provider teacher`
- B pairs (filter_off)      : `python data/b-hypothesis/build_pairs.py --input data/alpha-beta/blind-rollout/result/filter_off.jsonl --output data/paper/filter_off.jsonl --no-coupling-filter`
- B pairs (filter_on)       : `python data/b-hypothesis/build_pairs.py --input data/alpha-beta/blind-rollout/result/filter_off.jsonl --output data/paper/filter_on.jsonl --coupling-filter`

# train

- all variants             : `TRAIN_VARIANTS=core,aux,all,rw TRAIN_NUM_GPUS=3 ./train/train.sh`
- from prebuilt pairs       : `TRAIN_TRAJECTORY_DIR= DATA_DIR=data/paper TRAIN_VARIANTS=core,aux,all,rw ./train/train.sh`
- add to a session          : `RUN_ID=<id> TRAIN_VARIANTS=filter_off,filter_on TRAIN_AUTO_MERGE=false ./train/train.sh`
- one variant               : `python -m train.dpo_lora --trajectories data/alpha-beta/blind-rollout/result/a_beta_core.jsonl --out core --tensorboard`
- one variant (pairs)       : `python -m train.dpo_lora --pairs data/paper/a_beta_core.jsonl --out core`
- resume                    : `python -m train.dpo_lora --resume --out core`
- merge aux+all             : `python train/specialist-merge/merge_adapters.py --checkpoint-dir runs/<RUN_ID>/lora`

# eval

- full pipeline             : `./eval/eval.sh --run-id <RUN_ID>`
- rollout                   : `./eval/rollout/rollout.sh --variants base,core,aux,all,rw,merge --episodes 12`
- metrics                   : `./eval/metric/metrics.sh`
- tables                    : `./eval/table/tables.sh`
- paper tables 1–7          : `python -m eval.table.run_paper_tables --variants base,core,aux,all,rw,merge,filter_on,filter_off`
- one table                 : `python -m eval.table.run_table --table 1`
- one table (paper nums)    : `python -m eval.table.run_table --table 1 --compare-reference`
# check
