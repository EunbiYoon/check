"""Evaluation progress logging to terminal + file."""

from __future__ import annotations

import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class EvalLogger:
  """Print progress immediately and append the same lines to a log file."""

  def __init__(
    self,
    log_path: Path | None = None,
    enabled: bool = True,
    *,
    append: bool = False,
    label: str | None = None,
  ) -> None:
    self.enabled = enabled
    self.log_path = log_path
    self.label = label
    self._step = 0
    self._infer_count = 0
    self._t0 = time.time()
    if self.enabled and self.log_path:
      self.log_path.parent.mkdir(parents=True, exist_ok=True)
      if append and self.log_path.is_file():
        from eval.rollout.resume import infer_count_from_log

        self._infer_count = infer_count_from_log(self.log_path)
        header = f"=== eval resumed {self._ts()} (infer #{self._infer_count}) ===\n"
        with self.log_path.open("a", encoding="utf-8") as f:
          f.write(header)
      else:
        header = f"=== eval log started {self._ts()} ===\n"
        self.log_path.write_text(header, encoding="utf-8")

  def _ts(self) -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

  def log(self, msg: str) -> None:
    if not self.enabled:
      return
    label = f" [{self.label}]" if self.label else ""
    line = f"[{self._ts()}]{label} {msg}"
    print(line, flush=True)
    if self.log_path:
      # A long-running rollout may coexist with cleanup or layout migration of
      # the eval directory. Recreate the parent instead of crashing after the
      # expensive model inference has already completed.
      self.log_path.parent.mkdir(parents=True, exist_ok=True)
      with self.log_path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")

  def game_start(self, game_idx: int, n_games: int, game: str, n_episodes: int) -> None:
    self.log(f"[game {game_idx}/{n_games}] {game} - {n_episodes} episode(s)")

  def episode_start(self, game: str, episode: int, n_episodes: int, opponent: str) -> None:
    self.log(f"  [ep {episode}/{n_episodes}] {game} vs {opponent}")

  def episode_done(self, game: str, episode: int, cr: float, n_rounds: int) -> None:
    self.log(f"  [ep {episode} done] {game} CR={cr:.2f} rounds={n_rounds}")

  def infer_start(self, game: str, round_idx: int, horizon: int, opponent: str) -> int:
    self._infer_count += 1
    self._step += 1
    self.log(
      f"    [infer #{self._infer_count}] {game} "
      f"round {round_idx + 1}/{horizon} vs {opponent} ..."
    )
    return self._step

  def infer_done(self, step: int, game: str, action: Any, elapsed_s: float) -> None:
    self.log(f"    [infer #{self._infer_count} done] {game} action={action} ({elapsed_s:.1f}s)")

  def variant_start(self, variant: str, tag: str, mode: str, episodes: int) -> None:
    self.log(f"=== variant={variant} tag={tag} mode={mode} episodes={episodes} ===")

  def variant_done(self, variant: str, fc_id: float, fc_ho: float, cr_id: float, cr_ho: float) -> None:
    elapsed = time.time() - self._t0
    self.log(
      f"=== variant={variant} done in {elapsed / 60:.1f}min | "
      f"fc_id={fc_id:.2f} fc_ho={fc_ho:.2f} cr_id={cr_id:.2f} cr_ho={cr_ho:.2f} ==="
    )


class ProgressAgent:
  """Wrap an agent and log each decide() call."""

  def __init__(self, inner, logger: EvalLogger, opponent: str = "") -> None:
    self.inner = inner
    self.logger = logger
    self.opponent = opponent

  def decide(self, obs: dict[str, Any]) -> tuple[str, Any]:
    game = str(obs.get("game", "?"))
    rnd = int(obs.get("round", 0))
    horizon = int(obs.get("horizon", 1))
    step = self.logger.infer_start(game, rnd, horizon, self.opponent)
    t0 = time.perf_counter()
    reasoning, action = self.inner.decide(obs)
    self.logger.infer_done(step, game, action, time.perf_counter() - t0)
    return reasoning, action
