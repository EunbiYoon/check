# SAL pipeline

모든 명령은 저장소 루트에서 실행한다. GPU가 3개 이상 보이는 단일 노드를
기준으로 하며, 각 단계가 끝난 후 다음 단계를 실행한다.

## 1. Blind rollout

```bash
./data/blind-rollout/shard.sh "$node"
./data/blind-rollout/shard.sh merge
```
출력: `data/blind-rollout/result/a_beta_{core,aux,all,rw,fitler_off,filter_on}.jsonl`

## 2. A+beta pairs

```bash
CUDA_VISIBLE_DEVICES=0,1,2 A1_VARIANTS=core,aux,all,rw A1_INPUT_DIR=data/blind-rollout/result A1_OUTPUT_DIR=data/alpha-beta/result ./data/alpha-beta/algorithm1.sh
```

출력: `data/alpha-beta/result/a_beta_{core,aux,all,rw}.jsonl`

## 3. Hypothesis B pairs

```bash
CUDA_VISIBLE_DEVICES=0,1,2 HB_VARIANTS=filter_on,filter_off HB_RESULT_DIR=data/b-hypothesis/result ./data/b-hypothesis/hypothesis_b.sh
```

출력: `data/b-hypothesis/result/b_filter_{on,off}.jsonl`

## 4. DPO training

```bash
RUN_ID=paper DATA_DIR=data/alpha-beta/result TRAIN_VARIANTS=core,aux,all,rw TRAIN_NUM_GPUS=3 TRAIN_AUTO_MERGE=false ./train/dpo-lora/train.sh
```

GPU 0은 `core` 완료 후 `rw`, GPU 1은 `aux`, GPU 2는 `all`을 학습한다.
출력: `runs/paper/lora/{core,aux,all,rw}`

Hypothesis B를 학습할 때는 같은 명령에서 `DATA_DIR=data/b-hypothesis/result`와
`TRAIN_VARIANTS=filter_on,filter_off`만 사용한다.
