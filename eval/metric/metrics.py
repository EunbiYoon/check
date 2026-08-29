"""Aggregate metrics for Tables 1-7."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from config import ALL_GAMES, GAME_ABBREV, HO_GAMES, ID_GAMES, OPPONENT_AXIS
from eval.metric.coupling import coupling_for_round
from eval.rollout.games import EpisodeResult, run_game_batch
from eval.metric.exploitability import exploitability_2x2
from eval.rollout.progress import EvalLogger, ProgressAgent
from eval.rollout.resume import ParsedEpisode


@dataclass
class EvalMetrics:
    fc_id: float = 0.0
    fc_ho: float = 0.0
    ac_id: float = 0.0
    cr_id: float = 0.0
    cr_ho: float = 0.0
    per_game_cr: dict[str, float] = field(default_factory=dict)
    per_axis_cr: dict[str, float] = field(default_factory=dict)
    per_game_fc: dict[str, float | None] = field(default_factory=dict)
    per_game_ac: dict[str, float | None] = field(default_factory=dict)
    per_game_exploitability: dict[str, float | None] = field(default_factory=dict)
    episodes: list[EpisodeResult] = field(default_factory=list)


def _axis_for_opponent(name: str) -> str | None:
    for axis, members in OPPONENT_AXIS.items():
        if name in members:
            return axis
    return None


def _accumulate_episode_cr(
    game: str,
    cr: float,
    opponent: str | None,
    per_game: dict[str, list[float]],
    axis_cr: dict[str, list[float]],
) -> None:
    per_game[game].append(cr)
    if opponent:
        axis = _axis_for_opponent(opponent)
        if axis:
            bucket = f"{'ID' if game in ID_GAMES else 'HO'} {axis.title()}"
            axis_cr[bucket].append(cr)


def _accumulate_episode_metrics(
    game: str,
    ep: EpisodeResult,
    per_game: dict[str, list[float]],
    axis_cr: dict[str, list[float]],
    game_fc_num: dict[str, int],
    game_fc_den: dict[str, int],
    game_ac_num: dict[str, int],
    game_ac_den: dict[str, int],
    game_exp: dict[str, list[float]],
    fc_id_num: int,
    fc_id_den: int,
    fc_ho_num: int,
    fc_ho_den: int,
) -> tuple[int, int, int, int]:
    per_game[game].append(ep.cumulative_reward)

    for rnd in ep.rounds:
        if rnd.reasoning:
            c = coupling_for_round(rnd.reasoning, str(rnd.agent_action))
            if c.parsable:
                game_ac_den[game] += 1
                game_ac_num[game] += int(c.coupled)

    final = ep.final_round
    if final and final.reasoning:
        c = coupling_for_round(final.reasoning, str(final.agent_action))
        if c.parsable:
            game_fc_den[game] += 1
            game_fc_num[game] += int(c.coupled)
            if game in ID_GAMES:
                fc_id_den += 1
                fc_id_num += int(c.coupled)
            else:
                fc_ho_den += 1
                fc_ho_num += int(c.coupled)

    if final:
        exp = exploitability_2x2(game, str(final.agent_action))
        if exp is not None:
            game_exp[game].append(exp)

    axis = _axis_for_opponent(ep.opponent)
    if axis:
        bucket = f"{'ID' if game in ID_GAMES else 'HO'} {axis.title()}"
        axis_cr[bucket].append(ep.cumulative_reward)

    return fc_id_num, fc_id_den, fc_ho_num, fc_ho_den


def evaluate_agent(
    agent,
    n_episodes: int = 12,
    seed_base: int = 0,
    logger: EvalLogger | None = None,
    games: list[str] | None = None,
    prior_episodes: dict[str, list[ParsedEpisode]] | None = None,
) -> EvalMetrics:
    per_game: dict[str, list[float]] = defaultdict(list)
    fc_id_num = fc_id_den = fc_ho_num = fc_ho_den = 0
    axis_cr: dict[str, list[float]] = defaultdict(list)
    episodes: list[EpisodeResult] = []

    # Per-game coupling tallies
    game_fc_num: dict[str, int] = defaultdict(int)
    game_fc_den: dict[str, int] = defaultdict(int)
    game_ac_num: dict[str, int] = defaultdict(int)
    game_ac_den: dict[str, int] = defaultdict(int)
    game_exp: dict[str, list[float]] = defaultdict(list)

    games_to_run = set(games) if games else set(ALL_GAMES)
    prior = prior_episodes or {}

    for gi, game in enumerate(ALL_GAMES, start=1):
        prior_list = prior.get(game, [])
        skip_eps = {p.episode for p in prior_list}

        for p in prior_list:
            _accumulate_episode_cr(game, p.cr, p.opponent, per_game, axis_cr)

        if game not in games_to_run:
            continue

        if len(skip_eps) >= n_episodes:
            if logger:
                logger.game_start(gi, len(ALL_GAMES), game, n_episodes)
                logger.log(f"  [skip] {game}: all {n_episodes} episode(s) already complete")
            continue

        if logger:
            logger.game_start(gi, len(ALL_GAMES), game, n_episodes)
            if skip_eps:
                logger.log(f"  [resume] {game}: skipping {len(skip_eps)} episode(s), running {n_episodes - len(skip_eps)}")

        results = run_game_batch(
            game, agent, n_episodes, seed_base, logger=logger, skip_episodes=skip_eps
        )
        episodes.extend(results)
        for ep in results:
            fc_id_num, fc_id_den, fc_ho_num, fc_ho_den = _accumulate_episode_metrics(
                game,
                ep,
                per_game,
                axis_cr,
                game_fc_num,
                game_fc_den,
                game_ac_num,
                game_ac_den,
                game_exp,
                fc_id_num,
                fc_id_den,
                fc_ho_num,
                fc_ho_den,
            )

    cr_id_vals = [r for g in ID_GAMES for r in per_game[g]]
    cr_ho_vals = [r for g in HO_GAMES for r in per_game[g]]

    def _rate(num: dict, den: dict, g: str) -> float | None:
        if den.get(g, 0) == 0:
            return None
        return num[g] / den[g]

    def _mean_cr(g: str) -> float | None:
        if not per_game[g]:
            return None
        return sum(per_game[g]) / len(per_game[g])

    return EvalMetrics(
        fc_id=fc_id_num / fc_id_den if fc_id_den else 0.0,
        fc_ho=fc_ho_num / fc_ho_den if fc_ho_den else 0.0,
        ac_id=(
            sum(game_ac_num[g] for g in ID_GAMES) / sum(game_ac_den[g] for g in ID_GAMES)
            if sum(game_ac_den[g] for g in ID_GAMES)
            else 0.0
        ),
        cr_id=sum(cr_id_vals) / len(cr_id_vals) if cr_id_vals else 0.0,
        cr_ho=sum(cr_ho_vals) / len(cr_ho_vals) if cr_ho_vals else 0.0,
        per_game_cr={g: _mean_cr(g) for g in ALL_GAMES},
        per_axis_cr={k: sum(v) / len(v) for k, v in axis_cr.items()},
        per_game_fc={g: _rate(game_fc_num, game_fc_den, g) for g in ALL_GAMES},
        per_game_ac={g: _rate(game_ac_num, game_ac_den, g) for g in ALL_GAMES},
        per_game_exploitability={
            g: (sum(game_exp[g]) / len(game_exp[g]) if game_exp[g] else None)
            for g in ALL_GAMES
        },
        episodes=episodes,
    )


def metrics_to_dict(m: EvalMetrics) -> dict:
    return {
        "fc_id": m.fc_id,
        "fc_ho": m.fc_ho,
        "ac_id": m.ac_id,
        "cr_id": m.cr_id,
        "cr_ho": m.cr_ho,
        "per_game_cr": m.per_game_cr,
        "per_axis_cr": m.per_axis_cr,
        "per_game_fc": {k: v for k, v in m.per_game_fc.items()},
        "per_game_ac": {k: v for k, v in m.per_game_ac.items()},
        "per_game_exploitability": {k: v for k, v in m.per_game_exploitability.items()},
    }


def aggregate_episode_results(episodes: list[EpisodeResult]) -> EvalMetrics:
    """Aggregate already-persisted rollout episodes without running a model."""
    per_game: dict[str, list[float]] = defaultdict(list)
    axis_cr: dict[str, list[float]] = defaultdict(list)
    game_fc_num: dict[str, int] = defaultdict(int)
    game_fc_den: dict[str, int] = defaultdict(int)
    game_ac_num: dict[str, int] = defaultdict(int)
    game_ac_den: dict[str, int] = defaultdict(int)
    game_exp: dict[str, list[float]] = defaultdict(list)
    fc_id_num = fc_id_den = fc_ho_num = fc_ho_den = 0

    for ep in episodes:
        fc_id_num, fc_id_den, fc_ho_num, fc_ho_den = _accumulate_episode_metrics(
            ep.game,
            ep,
            per_game,
            axis_cr,
            game_fc_num,
            game_fc_den,
            game_ac_num,
            game_ac_den,
            game_exp,
            fc_id_num,
            fc_id_den,
            fc_ho_num,
            fc_ho_den,
        )

    cr_id = [value for game in ID_GAMES for value in per_game[game]]
    cr_ho = [value for game in HO_GAMES for value in per_game[game]]

    def rate(num: dict[str, int], den: dict[str, int], game: str) -> float | None:
        return num[game] / den[game] if den[game] else None

    return EvalMetrics(
        fc_id=fc_id_num / fc_id_den if fc_id_den else 0.0,
        fc_ho=fc_ho_num / fc_ho_den if fc_ho_den else 0.0,
        ac_id=(
            sum(game_ac_num[g] for g in ID_GAMES) / sum(game_ac_den[g] for g in ID_GAMES)
            if sum(game_ac_den[g] for g in ID_GAMES)
            else 0.0
        ),
        cr_id=sum(cr_id) / len(cr_id) if cr_id else 0.0,
        cr_ho=sum(cr_ho) / len(cr_ho) if cr_ho else 0.0,
        per_game_cr={
            game: sum(per_game[game]) / len(per_game[game]) if per_game[game] else None
            for game in ALL_GAMES
        },
        per_axis_cr={key: sum(values) / len(values) for key, values in axis_cr.items()},
        per_game_fc={game: rate(game_fc_num, game_fc_den, game) for game in ALL_GAMES},
        per_game_ac={game: rate(game_ac_num, game_ac_den, game) for game in ALL_GAMES},
        per_game_exploitability={
            game: sum(game_exp[game]) / len(game_exp[game]) if game_exp[game] else None
            for game in ALL_GAMES
        },
        episodes=episodes,
    )
