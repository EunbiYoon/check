"""Resume interrupted eval rollouts from progress logs."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from config import ALL_GAMES

_EP_START = re.compile(r"\[ep (\d+)/\d+\] ([\w-]+) vs (\w+)")
_EP_DONE = re.compile(r"\[ep (\d+) done\] ([\w-]+) CR=([\d.]+)")
_VARIANT = re.compile(r"=== variant=(\w+)")
_INFER = re.compile(r"\[infer #(\d+)\]")


@dataclass(frozen=True)
class ParsedEpisode:
    episode: int
    cr: float
    opponent: str | None


def parse_eval_log(path: Path, variant: str | None = None) -> dict[str, list[ParsedEpisode]]:
    """Parse completed episodes from an eval progress log."""
    if not path.is_file():
        return {}

    per_game: dict[str, list[ParsedEpisode]] = {}
    opponents: dict[tuple[str, int], str] = {}
    active = variant is None

    for line in path.read_text(encoding="utf-8").splitlines():
        vm = _VARIANT.search(line)
        if vm:
            active = variant is None or vm.group(1) == variant
            continue
        if not active:
            continue

        m = _EP_START.search(line)
        if m:
            ep_num, game, opp = int(m.group(1)), m.group(2), m.group(3)
            opponents[(game, ep_num)] = opp
            continue

        m = _EP_DONE.search(line)
        if m:
            ep_num, game, cr = int(m.group(1)), m.group(2), float(m.group(3))
            per_game.setdefault(game, []).append(
                ParsedEpisode(episode=ep_num, cr=cr, opponent=opponents.get((game, ep_num)))
            )

    for game, eps in per_game.items():
        deduped = {p.episode: p for p in eps}
        per_game[game] = [deduped[k] for k in sorted(deduped)]
    return per_game


def infer_count_from_log(path: Path) -> int:
    if not path.is_file():
        return 0
    return max(
        (int(m.group(1)) for line in path.read_text(encoding="utf-8").splitlines() for m in [_INFER.search(line)] if m),
        default=0,
    )


def parse_games_arg(raw: str | None) -> list[str] | None:
    if not raw:
        return None
    games = [g.strip() for g in raw.split(",") if g.strip()]
    unknown = [g for g in games if g not in ALL_GAMES]
    if unknown:
        raise ValueError(f"Unknown game(s): {', '.join(unknown)}. Valid: {', '.join(ALL_GAMES)}")
    return games


def resume_summary(prior: dict[str, list[ParsedEpisode]], n_episodes: int) -> str:
    if not prior:
        return "0 episodes"
    n_done = sum(len(eps) for eps in prior.values())
    complete_games = sum(1 for eps in prior.values() if len(eps) >= n_episodes)
    return f"{n_done} episode(s), {complete_games} game(s) fully complete"
