"""Prompt templates aligned with DPO pair format."""

from __future__ import annotations

from config import BOS_PAYOFFS, MP_PAYOFFS, PD_VARIANTS, STAG_PAYOFFS


def _pd_lines(game: str) -> list[str]:
    p = PD_VARIANTS[game]
    return [
        f"  If you play C and opp plays C: ({p.R}, {p.R})",
        f"  If you play C and opp plays D: ({p.S}, {p.T})",
        f"  If you play D and opp plays C: ({p.T}, {p.S})",
        f"  If you play D and opp plays D: ({p.P}, {p.P})",
    ]


def _format_history(history: list[dict]) -> str:
    if not history:
        return "(none yet — first round)."
    lines = []
    for i, row in enumerate(history, start=1):
        lines.append(
            f"  round {i}: actions={row['actions']} stage_rewards={row['rewards']}"
        )
    return "\n".join(lines)


def repeated_pd_prompt(game: str, *, round_idx: int, horizon: int, history: list[dict]) -> str:
    lines = [
        f"Game: {game} repeated for {horizon} rounds.",
        "You are p0. Opponents: ['p1'].",
        "Stage-game payoff table (your payoff, opponent payoff):",
        *_pd_lines(game),
        f"History: {_format_history(history)}",
        "Your legal actions: ['C', 'D']",
        "Choose one action. Remember the game continues.",
    ]
    if round_idx > 1:
        lines.insert(0, f"Round {round_idx}/{horizon}.")
    return "\n".join(lines)


def stag_hunt_prompt(*, round_idx: int, horizon: int, history: list[dict]) -> str:
    lines = [
        f"Game: stag_hunt repeated for {horizon} rounds.",
        "You are p0. Opponents: ['p1'].",
        "Stage-game payoff table (your payoff, opponent payoff):",
        "  If you play Stag and opp plays Stag: (4, 4)",
        "  If you play Stag and opp plays Hare: (0, 3)",
        "  If you play Hare and opp plays Stag: (3, 0)",
        "  If you play Hare and opp plays Hare: (2, 2)",
        f"History: {_format_history(history)}",
        "Your legal actions: ['Stag', 'Hare']",
        "Choose one action. Remember the game continues.",
    ]
    if round_idx > 1:
        lines.insert(0, f"Round {round_idx}/{horizon}.")
    return "\n".join(lines)


def bos_prompt() -> str:
    return (
        "Game: battle_of_sexes (one-shot simultaneous-move, 2 players).\n"
        "You are p0. Your opponent is p1.\n"
        "Payoff table (your payoff, opponent payoff):\n"
        "  If you play Opera and opp plays Opera: (2, 1)\n"
        "  If you play Opera and opp plays Football: (0, 0)\n"
        "  If you play Football and opp plays Opera: (0, 0)\n"
        "  If you play Football and opp plays Football: (1, 2)\n"
        "Your legal actions: ['Opera', 'Football']\n"
        "Choose one action from the list above."
    )


def matching_pennies_prompt() -> str:
    return (
        "Game: matching_pennies (one-shot simultaneous-move, 2 players).\n"
        "You are p0. Your opponent is p1.\n"
        "Payoff table (your payoff, opponent payoff):\n"
        "  If you play H and opp plays H: (1, -1)\n"
        "  If you play H and opp plays T: (-1, 1)\n"
        "  If you play T and opp plays H: (-1, 1)\n"
        "  If you play T and opp plays T: (1, -1)\n"
        "Your legal actions: ['H', 'T']\n"
        "Choose one action from the list above."
    )


def negotiation_prompt(*, weights: tuple[int, int, int] = (3, 4, 3)) -> str:
    wa, wb, wc = weights
    return (
        "Game: multi-issue bargaining (negotiation), one-shot simultaneous move.\n"
        "You are p1. Issues: a, b, c. Capacity per issue: 4 units.\n"
        f"Your private weights (payoff per unit claimed on each issue): a={wa}, b={wb}, c={wc} (sum={wa+wb+wc}).\n"
        "Both players simultaneously propose integer allocations [a_units, b_units, c_units],\n"
        "each component in [0, 4].\n"
        "If for every issue your claim + opponent's claim ≤ capacity, you receive\n"
        "weight[a]*your_a + weight[b]*your_b + weight[c]*your_c; otherwise both get 0.\n"
        "Choose one proposal from the legal list.\n"
        "Legal proposals (125 total), format [a,b,c]:\n"
        "[0,0,0], [0,0,1], [0,0,2], [0,0,3], [0,0,4], [0,1,0], [0,1,1], [0,1,2], [0,1,3], [0,1,4], [0,2,0], [0,2,1], ... (125 proposals total)\n"
        "Respond with <action>[a,b,c]</action> using integers in [0, 4] for each issue."
    )


def tic_tac_toe_prompt(board: list[str | None]) -> str:
    rows = []
    for r in range(3):
        cells = []
        for c in range(3):
            idx = r * 3 + c
            val = board[idx]
            cells.append(str(idx) if val is None else val)
        rows.append(" | ".join(cells))
    legal = [i for i, v in enumerate(board) if v is None]
    return (
        "Game: tic-tac-toe (3×3), sequential perfect-information zero-sum.\n"
        "You are p0 playing 'X'.\n"
        "Cells numbered 0–8 (row-major). Empty cells show their index.\n"
        "Terminal payoffs for you: win +1, draw 0, loss −1.\n"
        "Board:\n"
        f"{rows[0]}\n{rows[1]}\n{rows[2]}\n"
        f"Your turn. Legal moves: {legal}\n"
        "Choose one cell index from the legal list."
    )


def auction_prompt(*, value: int) -> str:
    return (
        "Game: sealed-bid first-price auction (one-shot).\n"
        f"You are the bidder. Your private value is {value}.\n"
        "Submit an integer bid in [0, 200].\n"
        "If your bid is strictly above the opponent bid, payoff = value - your_bid; else 0.\n"
        "Respond with <action>bid</action>."
    )


def divide_dollar_prompt() -> str:
    return (
        "Game: divide-the-dollar (one-shot).\n"
        "Propose your share as a float in [0.0, 1.0].\n"
        "If proposals sum to ≤ 1.0, you receive your proposed share; otherwise both get 0.\n"
        "Respond with <action>share</action>."
    )


def p_beauty_prompt() -> str:
    return (
        "Game: p-beauty contest (one-shot).\n"
        "Choose an integer guess in [0, 100].\n"
        "Payoff is higher when your guess is closer to 2/3 of the average guess.\n"
        "Respond with <action>guess</action>."
    )


def ipd_stage_prompt() -> str:
    p = PD_VARIANTS["ipd-stage"]
    return (
        "Game: ipd-stage (one-shot prisoner's dilemma).\n"
        "You are p0. Opponents: ['p1'].\n"
        "Stage-game payoff table (your payoff, opponent payoff):\n"
        f"  If you play C and opp plays C: ({p.R}, {p.R})\n"
        f"  If you play C and opp plays D: ({p.S}, {p.T})\n"
        f"  If you play D and opp plays C: ({p.T}, {p.S})\n"
        f"  If you play D and opp plays D: ({p.P}, {p.P})\n"
        "Your legal actions: ['C', 'D']\n"
        "Choose one action."
    )


def pd_payoff(game: str, a0: str, a1: str) -> tuple[float, float]:
    p = PD_VARIANTS[game]
    key = (a0, a1)
    table = {
        ("C", "C"): (p.R, p.R),
        ("C", "D"): (p.S, p.T),
        ("D", "C"): (p.T, p.S),
        ("D", "D"): (p.P, p.P),
    }
    return table[(a0, a1)]


def stag_payoff(a0: str, a1: str) -> tuple[float, float]:
    idx = {"Stag": 0, "Hare": 1}
    v = STAG_PAYOFFS[(idx[a0], idx[a1])]
    return float(v), float(STAG_PAYOFFS[(idx[a1], idx[a0])])


def bos_payoff(a0: str, a1: str) -> tuple[float, float]:
    idx = {"Opera": 0, "Football": 1}
    return BOS_PAYOFFS[(idx[a0], idx[a1])]


def mp_payoff(a0: str, a1: str) -> tuple[float, float]:
    idx = {"H": 0, "T": 1}
    return MP_PAYOFFS[(idx[a0], idx[a1])]
