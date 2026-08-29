# SAL pipeline

Run all commands from the repository root. These instructions assume a single
node with at least three visible GPUs. Complete each stage before starting the next.

## 1. Blind rollout

```bash
./data/blind-rollout/shard.sh "$node"
./data/blind-rollout/shard.sh merge
```
Output: `data/blind-rollout/result/a_beta_{core,aux,all,rw,filter_off,filter_on}.jsonl`

## 2. A+beta pairs

```bash
CUDA_VISIBLE_DEVICES=0,1,2 A1_VARIANTS=core,aux,all,rw A1_INPUT_DIR=data/blind-rollout/result A1_OUTPUT_DIR=data/alpha-beta/result ./data/alpha-beta/algorithm1.sh
```

Output: `data/alpha-beta/result/a_beta_{core,aux,all,rw}.jsonl`

## 3. Hypothesis B pairs

```bash
CUDA_VISIBLE_DEVICES=0,1,2 HB_VARIANTS=filter_on,filter_off HB_RESULT_DIR=data/b-hypothesis/result ./data/b-hypothesis/hypothesis_b.sh
```

Output: `data/b-hypothesis/result/b_filter_{on,off}.jsonl`

## 4. DPO training

```bash
RUN_ID=paper DATA_DIR=data/alpha-beta/result TRAIN_VARIANTS=core,aux,all,rw TRAIN_NUM_GPUS=3 TRAIN_AUTO_MERGE=false ./train/dpo-lora/train.sh
```

GPU 0 trains `core` followed by `rw`, GPU 1 trains `aux`, and GPU 2 trains `all`.
Output: `runs/paper/lora/{core,aux,all,rw}`

To train Hypothesis B, use the same command with
`DATA_DIR=data/b-hypothesis/result` and `TRAIN_VARIANTS=filter_on,filter_off`.
