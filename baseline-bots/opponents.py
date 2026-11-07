"""Stronger practice opponents (baseline-add-1 .. baseline-add-4): they look at their cards.

Each style is a function(obs, memory) -> Action, optionally with an end-of-hand function.
Used by baseline-add-N.py (HTTP servers, via addbase.py) and by train/selfplay.py (in-process),
so both always play the same way. They are practice opponents only, not part of the submission.

  add1  ValueBettor : solid. Calls with pot odds, raises pot only with strong hands, never bluffs.
  add2  Maniac      : raises pot ~45% of the time whatever it holds, calls/raises sensibly when facing a bet.
  add3  Nit         : plays only strong hands. Folds a lot (punishes calling too much, bluffs work).
  add4  Adaptive    : tracks how often the opponent folds. Opponent folds a lot -> bluffs hard, else value plays.
"""
import random

from bot.equity import mc_equity  # same evaluator, so the baselines see true hand strength
from game import Action

H, P, C, F = Action.RAISE_HALF, Action.RAISE_POT, Action.CHECK_CALL, Action.FOLD
SAMPLES = 30  # Monte Carlo samples per street (cached), enough for a practice opponent


def _rng(m):
    return m.setdefault("rng", random.Random())


def _eq(obs, m):
    key = (obs.hand_index, obs.street)
    if m.get("eq_key") != key:
        m["eq_key"], m["eq"] = key, mc_equity(obs.hole_cards, obs.board, SAMPLES, _rng(m))
    return m["eq"]


def _pick(obs, *prefs):
    """First legal action among the preferences, else CHECK_CALL."""
    for a in prefs:
        if a in obs.legal_actions:
            return a
    return C


def _need(obs):
    return obs.to_call / (obs.pot + obs.to_call) if obs.to_call else 0.0


def add1(obs, m):
    eq, need = _eq(obs, m), _need(obs)
    if obs.to_call:
        if eq < need + 0.03:
            return _pick(obs, F)
        return _pick(obs, P, C) if eq >= 0.62 else C
    return _pick(obs, P, H, C) if eq >= 0.7 else _pick(obs, H, C) if eq >= 0.55 else C


def add2(obs, m):
    eq, need = _eq(obs, m), _need(obs)
    if obs.to_call:
        if eq >= 0.6:
            return _pick(obs, P, C)
        return C if eq >= need else _pick(obs, F)
    return _pick(obs, P, C) if _rng(m).random() < 0.45 or eq >= 0.6 else C


def add3(obs, m):
    eq, need = _eq(obs, m), _need(obs)
    if obs.to_call:
        if eq < max(need + 0.1, 0.5):
            return _pick(obs, F)
        return _pick(obs, P, C) if eq >= 0.72 else C
    return _pick(obs, H, C) if eq >= 0.6 else C


def add4(obs, m):
    eq, need = _eq(obs, m), _need(obs)
    fold_rate = m.get("won_by_fold", 0) / max(1, m.get("hands", 0))
    bluffy = m.get("hands", 0) >= 30 and fold_rate > 0.3  # opponent gives up too often: attack
    if obs.to_call:
        if eq >= (0.55 if bluffy else 0.65):
            return _pick(obs, P, C)
        return C if eq >= need else _pick(obs, F)
    if eq >= 0.65 or (bluffy and _rng(m).random() < 0.6):
        return _pick(obs, P, C)
    return C


def add4_end(result, m):
    m["hands"] = m.get("hands", 0) + 1
    m["won_by_fold"] = m.get("won_by_fold", 0) + (result.winner == "me" and not result.went_to_showdown)


STYLES = {"add1": (add1, None), "add2": (add2, None), "add3": (add3, None), "add4": (add4, add4_end)}
NAMES = {"add1": "Add1-ValueBettor", "add2": "Add2-Maniac", "add3": "Add3-Nit", "add4": "Add4-Adaptive"}
