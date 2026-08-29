"""SARL hyperparameters and local overrides for this machine."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

PROJECT_ROOT = Path(__file__).resolve().parent


def _setup_hf_cache() -> None:
    """Use scratch for HF cache when home quota is tight."""
    import os

    user = os.environ.get("USER") or os.environ.get("USERNAME", "")
    if not user:
        return
    scratch_hf = Path(f"/scratch/workspace/eunbiyoon_umass_edu-paper/{user}/.cache/huggingface")
    if not scratch_hf.parent.parent.is_dir():
        return
    os.environ.setdefault("HF_HOME", str(scratch_hf))
    os.environ.setdefault("HF_HUB_CACHE", str(scratch_hf / "hub"))
    scratch_hf.mkdir(parents=True, exist_ok=True)
    (scratch_hf / "hub").mkdir(parents=True, exist_ok=True)


def _load_dotenv() -> None:
    """Load PROJECT_ROOT/.env; map hf_token -> HF_TOKEN for huggingface_hub."""
    import os

    _setup_hf_cache()

    env_path = PROJECT_ROOT / ".env"
    if not env_path.is_file():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        key, val = key.strip(), val.strip().strip('"').strip("'")
        val = os.path.expandvars(val)
        os.environ.setdefault(key, val)
    token = os.environ.get("HF_TOKEN") or os.environ.get("hf_token")
    if token:
        os.environ.setdefault("HF_TOKEN", token)
        os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", token)


_load_dotenv()


def _env_str(name: str, default: str) -> str:
    import os

    return os.environ.get(name, default)


def _env_int(name: str, default: int) -> int:
    return int(_env_str(name, str(default)))


def _env_seed(name: str, default: int) -> int:
    value = _env_str(name, str(default)).strip().lower()
    if value == "random":
        import secrets

        return secrets.randbelow(2_147_483_647)
    return int(value)


def _env_float(name: str, default: float) -> float:
    return float(_env_str(name, str(default)))


def _env_optional_int(name: str, default: int | None = None) -> int | None:
    value = _env_str(name, "" if default is None else str(default)).strip()
    return int(value) if value else None


def _env_bool(name: str, default: bool) -> bool:
    return _env_str(name, str(default)).lower() in {"1", "true", "yes", "on"}

DPO_DIR = PROJECT_ROOT / "dpo"
DPO_DATA_DIR = DPO_DIR / "data"
LORA_DIR = PROJECT_ROOT / "lora"
# Session run outputs live under history/<session>/{lora,eval}/.
# The pre-rename location was runs/; fall back to it if history/ is absent.
HISTORY_DIR = PROJECT_ROOT / "history"
RUNS_DIR = HISTORY_DIR if HISTORY_DIR.is_dir() or not (PROJECT_ROOT / "runs").is_dir() else PROJECT_ROOT / "runs"
# Legacy aliases — prefer RUNS_DIR + history.paths helpers
LORA_RUNS_DIR = RUNS_DIR
EVAL_RUNS_DIR = RUNS_DIR
LORA_3B_DIR = RUNS_DIR  # resolved per-session via RUN_ID in train/eval scripts
RESULTS_DIR = PROJECT_ROOT / "results"
EVAL_DIR = PROJECT_ROOT / "eval"
# Evaluation: 12 envs × EPISODES_PER_ENV episodes; ~3.5–4 h per variant on RTX 3060 12GB
EVAL_DEFAULT_VARIANTS = tuple(
    v.strip() for v in _env_str("EVAL_VARIANTS", "base,core").split(",") if v.strip()
)
EVAL_EST_SECONDS_PER_VARIANT = 3.75 * 3600  # wall-clock hint for ep=12, 0.5B 4-bit

# DPO generate config (Ollama): PD classic / tight / high-temptation
DPO_CONFIG_PD = DPO_DIR / "configs" / "pd.yaml"
DEFAULT_N_EPISODES = 1
PAPER_N_EPISODES = 1000
# Backward-compatible aliases
DPO_CONFIG_PAPER = DPO_CONFIG_PD
OLLAMA_BASE_URL = "http://127.0.0.1:11434"
OLLAMA_MAX_TOKENS_PD = 256
OLLAMA_FRONTIER_MODEL = "llama3.1:8b"  # Table 1 Llama 3.1–8B via local Ollama

# Qwen2.5-3B-Instruct, LoRA r=16 alpha=32 dropout=0.05, DPO beta=0.1
STUDENT_MODEL = _env_str("STUDENT_MODEL", "Qwen/Qwen2.5-3B-Instruct")
TEACHER_MODEL = _env_str("TEACHER_MODEL", "Qwen/Qwen2.5-7B-Instruct")

# Algorithm 1 line-2 blind rollout.
BLIND_MODEL = _env_str("BLIND_MODEL", TEACHER_MODEL)
BLIND_GPU = _env_str("BLIND_GPU", "0")
BLIND_VARIANTS = tuple(
    variant.strip() for variant in _env_str("BLIND_VARIANTS", "all").split(",")
    if variant.strip()
)
BLIND_GAMES = tuple(
    game.strip() for game in _env_str(
        "BLIND_GAMES",
        "pd-classic,pd-tight,pd-high-temptation,stag-hunt,bos,matching-pennies,"
        "negotiation,auction,ipd-stage",
    ).split(",") if game.strip()
)
BLIND_EPISODES_PER_COMBINATION = _env_int("BLIND_EPISODES_PER_COMBINATION", 10)
BLIND_SEED = _env_seed("BLIND_SEED", 42)
BLIND_MAX_NEW_TOKENS = _env_int("BLIND_MAX_NEW_TOKENS", 384)
BLIND_DO_SAMPLE = _env_bool("BLIND_DO_SAMPLE", True)
BLIND_TEMPERATURE = _env_float("BLIND_TEMPERATURE", 0.7)
BLIND_OUTPUT = _env_str(
    "BLIND_OUTPUT", "data/alpha-beta/blind-rollout/result/a_beta_all.jsonl"
)

# Compatibility aliases for evaluation/training modules. Both resolve to the
# student because the blind rollout policy and the LoRA target are identical.
STUDENT_MODEL = STUDENT_MODEL
PAPER_STUDENT_MODEL = STUDENT_MODEL

# Table 1 frontier references use the pretrained base, not -Instruct.
FRONTIER_STUDENT_MODELS = {
    "gemma_27b": "google/gemma-3-27b-it",
    "llama_70b": "meta-llama/Llama-3.1-70B",  # 3.3 base not published; Meta only ships 3.3-Instruct
    "llama_8b": "meta-llama/Llama-3.1-8B",
}

LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = _env_float("LORA_DROPOUT", 0.05)
LORA_TARGET_MODULES = ["q_proj", "v_proj"]  # matches working lora.py

PAPER_LORA_R = _env_int("LORA_R", 16)
PAPER_LORA_ALPHA = _env_int("LORA_ALPHA", 32)
PAPER_LORA_TARGET_MODULES = _env_str("LORA_TARGET", "all-linear")

DPO_BETA = _env_float("DPO_BETA", 0.1)
LEARNING_RATE = _env_float("LEARNING_RATE", 5e-5)
NUM_EPOCHS = _env_int("TRAIN_EPOCHS", 10)
MAX_TRAIN_STEPS = _env_optional_int("TRAIN_MAX_STEPS")
VARIANT_MAX_TRAIN_STEPS = {
    variant: _env_optional_int(f"TRAIN_MAX_STEPS_{variant.upper()}")
    for variant in ("filter_on", "filter_off", "core", "aux", "all", "rw")
}


def max_train_steps_for_variant(variant: str) -> int | None:
    """Return a variant-specific step cap, falling back to the shared cap."""
    variant_value = VARIANT_MAX_TRAIN_STEPS.get(variant)
    return MAX_TRAIN_STEPS if variant_value is None else variant_value


EPISODES_PER_ENV = _env_int("EVAL_EPISODES", 12)
EVAL_SEED = _env_int("EVAL_SEED", 42)
EVAL_MAX_TOKENS = _env_int("EVAL_MAX_TOKENS", 192)
EVAL_VARIANTS = _env_str("EVAL_VARIANTS", "base,core")
EVAL_GAMES = _env_str("EVAL_GAMES", "all")
USE_4BIT = _env_bool("USE_4BIT", True)
BNB_4BIT_QUANT_TYPE = _env_str("BNB_4BIT_QUANT_TYPE", "nf4")
BNB_4BIT_COMPUTE_DTYPE = _env_str("BNB_4BIT_COMPUTE_DTYPE", "float16")
GRADIENT_CHECKPOINTING = _env_bool("GRADIENT_CHECKPOINTING", True)
TRAIN_FP16 = _env_bool("FP16", False)
TRAIN_BF16 = _env_bool("BF16", False)
GENERATE_N_EPISODES = 1000  # DPO rollout count per environment
GENERATE_N_EPISODES_DEMO = 16

# Pipeline layout: dpo/runs/<timestamp>/ + dpo/data/*.jsonl | lora/<variant>/ adapters

ID_GAMES = [
    "pd-classic",
    "pd-tight",
    "pd-high-temptation",
    "stag-hunt",
    "negotiation",
    "bos",
    "matching-pennies",
    "tic-tac-toe",
]

HO_GAMES = [
    "auction",
    "divide-dollar",
    "p-beauty",
    "ipd-stage",
]

ALL_GAMES = ID_GAMES + HO_GAMES

VariantName = Literal[
    "base",
    "core",
    "aux",
    "all",
    "rw",
    "merge",
]


@dataclass
class PDParams:
    T: int
    R: int
    P: int
    S: int
    horizon: int = 10


PD_VARIANTS = {
    "pd-classic": PDParams(T=5, R=3, P=1, S=0),
    "pd-tight": PDParams(T=4, R=3, P=2, S=1),
    "pd-high-temptation": PDParams(T=8, R=3, P=1, S=0),
    "ipd-stage": PDParams(T=5, R=3, P=1, S=0, horizon=1),
}

STAG_PAYOFFS = {(0, 0): 4, (0, 1): 0, (1, 0): 3, (1, 1): 1}  # 0=Stag, 1=Hare

BOS_PAYOFFS = {
    (0, 0): (3, 2),  # Opera, Opera
    (0, 1): (0, 0),
    (1, 0): (0, 0),
    (1, 1): (2, 3),  # Football, Football
}

MP_PAYOFFS = {
    (0, 0): (1, -1),
    (0, 1): (-1, 1),
    (1, 0): (-1, 1),
    (1, 1): (1, -1),
}

OPPONENT_AXIS = {
    "stationary": ["always_cooperate", "always_defect"],
    "stochastic": ["random", "epsilon_greedy"],
    "strategic": ["tit_for_tat", "grim_trigger", "pavlov", "tit_for_two_tats", "generous_tft"],
}

# Table 1 reference rows used by --compare-reference.
PAPER_TABLE1 = {
    "base": {"fc_id": 0.77, "fc_ho": 0.56, "cr_id": 14.04, "cr_ho": -1.45},
    "core": {"fc_id": 0.61, "fc_ho": 0.00, "cr_id": 13.79, "cr_ho": 0.44},
    "aux": {"fc_id": 0.56, "fc_ho": 0.29, "cr_id": 13.86, "cr_ho": 0.68},
    "all": {"fc_id": 0.59, "fc_ho": 0.29, "cr_id": 14.34, "cr_ho": -0.27},
    "rw": {"fc_id": 0.68, "fc_ho": 0.46, "cr_id": 13.93, "cr_ho": 1.91},
    "merge": {"fc_id": 0.69, "fc_ho": 0.50, "cr_id": 13.92, "cr_ho": 1.75},
    "haiku": {"fc_id": 0.85, "fc_ho": 0.79, "cr_id": 14.60, "cr_ho": 2.05},
}

PAPER_TABLE2 = {
    "pd-classic": {"base": 21.96, "core": 22.08, "aux": 23.08, "all": 23.88, "rw": 21.67, "merge": 23.50, "haiku": 25.68},
    "pd-tight": {"base": 25.92, "core": 27.08, "aux": 26.21, "all": 26.21, "rw": 25.42, "merge": 25.92, "haiku": 27.25},
    "pd-high-temptation": {"base": 29.33, "core": 28.50, "aux": 29.17, "all": 30.42, "rw": 28.42, "merge": 29.17, "haiku": 27.92},
    "stag-hunt": {"base": 28.62, "core": 28.71, "aux": 28.38, "all": 30.54, "rw": 30.42, "merge": 27.88, "haiku": 27.79},
    "negotiation": {"base": 5.75, "core": 3.17, "aux": 3.46, "all": 3.08, "rw": 4.67, "merge": 4.04, "haiku": 7.54},
    "bos": {"base": 0.75, "core": 0.75, "aux": 0.62, "all": 0.62, "rw": 0.88, "merge": 0.88, "haiku": 0.62},
    "matching-pennies": {"base": 0.00, "core": 0.00, "aux": 0.00, "all": 0.00, "rw": 0.00, "merge": 0.00, "haiku": 0.00},
    "tic-tac-toe": {"base": 0.00, "core": 0.00, "aux": 0.00, "all": 0.00, "rw": 0.00, "merge": 0.00, "haiku": 0.00},
    "auction": {"base": -8.67, "core": -0.71, "aux": -0.10, "all": -3.96, "rw": 4.97, "merge": 4.18, "haiku": 5.82},
    "divide-dollar": {"base": 0.27, "core": 0.11, "aux": 0.22, "all": 0.34, "rw": 0.32, "merge": 0.21, "haiku": 0.25},
    "p-beauty": {"base": 0.50, "core": 0.50, "aux": 0.50, "all": 0.50, "rw": 0.50, "merge": 0.50, "haiku": 0.50},
    "ipd-stage": {"base": 2.08, "core": 1.88, "aux": 2.08, "all": 2.04, "rw": 1.83, "merge": 2.12, "haiku": 1.62},
}

PAPER_TABLE3 = {
    "ID Stationary": {"base": 15.60, "core": 16.19, "aux": 15.84, "all": 16.35, "rw": 16.40, "merge": 15.92, "haiku": 18.79},
    "ID Stochastic": {"base": 6.22, "core": 5.53, "aux": 6.66, "all": 6.21, "rw": 6.37, "merge": 6.34, "haiku": 6.83},
    "ID Strategic": {"base": 13.26, "core": 15.12, "aux": 15.25, "all": 16.35, "rw": 14.62, "merge": 15.54, "haiku": 21.13},
    "HO Stationary": {"base": 1.41, "core": 1.29, "aux": 1.48, "all": 1.48, "rw": 1.37, "merge": 1.49, "haiku": 1.10},
    "HO Stochastic": {"base": -3.33, "core": -0.07, "aux": 0.20, "all": -1.32, "rw": 2.23, "merge": 1.91, "haiku": 2.67},
    # Filled from evaluation after LoRA training.
    "HO Strategic": {"base": None, "core": None, "aux": None, "all": None, "rw": None, "merge": None, "haiku": None},
}

# Table 1 extra variants (Hypothesis B)
PAPER_TABLE1.update({
    "filter_on": {"fc_id": 0.66, "fc_ho": 0.36, "cr_id": 13.77, "cr_ho": -0.95},
    "filter_off": {"fc_id": 0.55, "fc_ho": 0.71, "cr_id": 13.51, "cr_ho": -0.23},
})

# All trainable LoRA variants used in Tables 1 and 5.
TRAINING_VARIANTS = {
    "filter_on": {"pairs": 388, "best_step": 60, "best_eval_loss": 0.3734, "data": "dpo/data/filter_on.jsonl"},
    "filter_off": {"pairs": 407, "best_step": 110, "best_eval_loss": 0.2721, "data": "dpo/data/filter_off.jsonl"},
    "core": {"pairs": 503, "best_step": 50, "best_eval_loss": 0.2973, "data": "dpo/data/a_beta_core.jsonl"},
    "aux": {"pairs": 613, "best_step": 80, "best_eval_loss": 0.3128, "data": "dpo/data/a_beta_aux.jsonl"},
    "all": {"pairs": 1338, "best_step": 40, "best_eval_loss": 0.2278, "data": "dpo/data/a_beta_all.jsonl"},
    "rw": {"pairs": 1749, "best_step": 70, "best_eval_loss": 0.2454, "data": "dpo/data/a_beta_rw.jsonl"},
    "merge": {"pairs": 0, "best_step": None, "best_eval_loss": None, "data": None},
}

# Eq. 4 LoRA merge weight: W_merge = MERGE_ALPHA * W_AUX + (1 - MERGE_ALPHA) * W_ALL
MERGE_ALPHA = _env_float("MERGE_ALPHA", 0.5)

# Appendix E full training hyperparameters
MAX_SEQ_LENGTH = _env_int("MAX_LENGTH", 12288)
LOCAL_MAX_SEQ_LENGTH = 8192  # RTX 3060 12GB: fits ~7.7k-token DPO pairs
PER_DEVICE_BATCH = _env_int("TRAIN_BATCH_SIZE", 1)
GRADIENT_ACCUMULATION = _env_int("GRAD_ACCUM", 8)
LOCAL_GRADIENT_ACCUMULATION = 4  # lower VRAM peak during DPO (chosen+rejected)
WARMUP_RATIO = _env_float("WARMUP_RATIO", 0.05)
LR_SCHEDULER = _env_str("LR_SCHEDULER", "cosine")
LOGGING_STEPS = _env_int("LOGGING_STEPS", 10)
SAVE_STEPS = _env_int("SAVE_STEPS", 10)
PAPER_SAVE_TOTAL_LIMIT = _env_int("SAVE_TOTAL_LIMIT", 100)

TABLE1_VARIANTS = [
    "base", "filter_on", "filter_off", "core", "aux", "all", "rw", "merge",
]
PAPER_EVAL_VARIANTS = TABLE1_VARIANTS  # Table 1–3 evaluation set
TABLE2_VARIANTS = ["base", "core", "aux", "all", "rw", "merge", "haiku"]
TABLE3_VARIANTS = ["base", "core", "aux", "all", "rw", "merge", "haiku"]

# Table 4 environment abbreviations.
GAME_ABBREV = {
    "pd-classic": "pd-c",
    "pd-tight": "pd-t",
    "pd-high-temptation": "pd-h",
    "stag-hunt": "stag",
    "negotiation": "nego",
    "bos": "bos",
    "matching-pennies": "mp",
    "tic-tac-toe": "ttt",
    "auction": "auct",
    "divide-dollar": "dd",
    "p-beauty": "p-b",
    "ipd-stage": "ipd",
}
ABBREV_TO_GAME = {v: k for k, v in GAME_ABBREV.items()}

# Table 4 reference (final-round coupling fc per model x environment).
PAPER_TABLE4 = {
    "haiku": {"pd-c": 0.87, "pd-t": 0.90, "pd-h": 0.91, "stag": 0.89, "nego": 0.00, "bos": 1.00, "mp": 1.00, "ttt": None, "auct": 0.00, "dd": None, "p-b": 0.00, "ipd": 0.92},
    "llama_8b": {"pd-c": 0.90, "pd-t": 0.43, "pd-h": 1.00, "stag": 0.43, "nego": 0.00, "bos": 1.00, "mp": 0.71, "ttt": 0.00, "auct": None, "dd": 0.00, "p-b": None, "ipd": 0.67},
    "gemma_27b": {"pd-c": 0.38, "pd-t": 0.75, "pd-h": 0.83, "stag": 0.91, "nego": 0.00, "bos": 1.00, "mp": 0.91, "ttt": 0.00, "auct": None, "dd": 0.00, "p-b": None, "ipd": 1.00},
    "llama_70b": {"pd-c": 0.60, "pd-t": 0.33, "pd-h": 0.80, "stag": 0.50, "nego": 0.00, "bos": 1.00, "mp": 1.00, "ttt": 0.00, "auct": 0.00, "dd": 0.00, "p-b": 0.00, "ipd": 0.75},
    "base": {"pd-c": 1.00, "pd-t": 0.80, "pd-h": 0.83, "stag": 0.75, "nego": None, "bos": 0.78, "mp": 0.75, "ttt": 0.00, "auct": None, "dd": 0.00, "p-b": 0.00, "ipd": 1.00},
    "abeta_on": {"pd-c": 0.83, "pd-t": 0.33, "pd-h": 0.50, "stag": 1.00, "nego": 0.00, "bos": 1.00, "mp": 0.60, "ttt": None, "auct": 0.00, "dd": 0.00, "p-b": 0.00, "ipd": 0.67},
    "abeta_off": {"pd-c": 0.25, "pd-t": 0.33, "pd-h": 0.33, "stag": 0.00, "nego": 0.00, "bos": 0.88, "mp": 0.88, "ttt": 0.00, "auct": None, "dd": None, "p-b": 0.00, "ipd": 0.83},
}

# Table 5 reference (reasoning-action coupling: fc/ac by ID/HO).
PAPER_TABLE5_COUPLING = {
    "core": {"fc_id": 0.61, "ac_id": 0.74, "fc_ho": 0.00},
    "aux": {"fc_id": 0.56, "ac_id": 0.72, "fc_ho": 0.29},
    "all": {"fc_id": 0.59, "ac_id": 0.65, "fc_ho": 0.29},
    "rw": {"fc_id": 0.68, "ac_id": 0.70, "fc_ho": 0.46},
    "merge": {"fc_id": 0.69, "ac_id": 0.74, "fc_ho": 0.50},
    "haiku": {"fc_id": 0.85, "ac_id": 0.57, "fc_ho": 0.79},
    "gemma_27b": {"fc_id": 0.74, "ac_id": 0.80, "fc_ho": 0.43},
    "llama_70b": {"fc_id": 0.69, "ac_id": 0.52, "fc_ho": 0.30},
}

# Table 5 reference (training manifest with legacy numbering).
PAPER_TABLE5 = {
    v: {"pairs": TRAINING_VARIANTS[v]["pairs"], "best_step": TRAINING_VARIANTS[v]["best_step"], "best_eval_loss": TRAINING_VARIANTS[v]["best_eval_loss"]}
    for v in TRAINING_VARIANTS
}

# Table 6 stag-hunt anti-coordination example.
PAPER_TABLE6 = [
    {"round": 1, "tft": "Stag", "blind": "Hare", "oracle": "Stag", "helps": True},
    {"round": 2, "tft": "Hare", "blind": "Stag", "oracle": "Hare", "helps": True},
    {"round": 3, "tft": "Stag", "blind": "Hare", "oracle": "Stag", "helps": True},
    {"round": 6, "tft": "Hare", "blind": "Stag", "oracle": "Hare", "helps": True},
    {"round": 7, "tft": "Stag", "blind": "Hare", "oracle": "Stag", "helps": True},
]

# Table 7 reference (Haiku 4.5 per-environment ac, fc, exploitability).
PAPER_TABLE7 = {
    "pd-classic": {"ac": 0.44, "fc": 0.91, "exploitability": 0.73},
    "pd-tight": {"ac": 0.47, "fc": 0.90, "exploitability": 0.72},
    "pd-high-temptation": {"ac": 0.54, "fc": 0.91, "exploitability": 0.58},
    "stag-hunt": {"ac": 0.85, "fc": 0.89, "exploitability": 0.27},
    "negotiation": {"ac": 0.00, "fc": 0.00, "exploitability": None},
    "bos": {"ac": 1.00, "fc": 1.00, "exploitability": 0.00},
    "matching-pennies": {"ac": 1.00, "fc": 1.00, "exploitability": 1.00},
    "tic-tac-toe": {"ac": 0.00, "fc": None, "exploitability": None},
    "auction": {"ac": 0.00, "fc": 0.00, "exploitability": None},
    "divide-dollar": {"ac": None, "fc": None, "exploitability": None},
    "p-beauty": {"ac": 0.00, "fc": 0.00, "exploitability": None},
    "ipd-stage": {"ac": 0.92, "fc": 0.92, "exploitability": 0.00},
}

VARIANT_LABELS = {
    "base": "Qwen 2.5-3B base",
    "filter_on": "Qwen 2.5-3B + DPO (filter-on)",
    "filter_off": "Qwen 2.5-3B + DPO (filter-off)",
    "core": "Qwen 2.5-3B + A+b-CORE",
    "aux": "Qwen 2.5-3B + A+b-AUX",
    "all": "Qwen 2.5-3B + A+b-ALL",
    "rw": "Qwen 2.5-3B + A+b-RW",
    "merge": "Qwen 2.5-3B + A+b-MERGE",
    "haiku": "Bedrock Haiku 4.5",
}
