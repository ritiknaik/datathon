"""
Tests for YOUR bot (agent.py). Run them from the bundle folder:

    python -m pytest -q

They talk to your bot the same way the judge does (through its web routes,
but without starting a real server), so a passing run means the judge can
play with your bot. Add your own tests at the bottom!
"""

import importlib.util
import os
import random
import sys

import pytest
from fastapi.testclient import TestClient

# Works whether this file sits next to agent.py or inside a tests/ folder.
HERE = os.path.dirname(os.path.abspath(__file__))
BUNDLE = HERE if os.path.isfile(os.path.join(HERE, "game.py")) else os.path.dirname(HERE)
sys.path.insert(0, BUNDLE)  # so agent.py can import game.py and evaluator.py

sys.path.insert(0, os.path.join(BUNDLE, "baseline-bots"))  # stronger baselines (opponents.py)
from game import Action, AgentConfig, Game, make_deal  # noqa: E402
from judge_engine import coerce_action  # noqa: E402  (reads answers exactly like the judge)

TOKEN = "test-token"
AUTH = {"X-Agent-Token": TOKEN}


def load_agent():
    """Load a fresh copy of agent.py (its own MEMORY and globals)."""
    spec = importlib.util.spec_from_file_location("agent_under_test", os.path.join(BUNDLE, "agent.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def bot():
    """(agent module, test client) with the tournament's token check switched on."""
    module = load_agent()
    module.TOKEN = TOKEN
    return module, TestClient(module.app)


def start_half(client, half=1, seat=0):
    """POST /new-half the way the judge does."""
    body = {"half": half, "seat": seat, "config": AgentConfig().to_dict()}
    return client.post("/new-half", json=body, headers=AUTH)


def play_hands(client, num_hands, seed="test", raise_rate=0.3):
    """Play hands where YOUR bot is seat 0 and a random opponent is seat 1.

    Every decision goes through /act exactly like in a real match, and
    finished hands are delivered with the next /act (like the judge does).
    Returns the list of every action your bot chose.
    """
    rng = random.Random(seed)
    pending = []      # finished hands waiting to be delivered
    chosen = []
    for hand in range(num_hands):
        game = Game(make_deal(seed, hand), button_seat=hand % 2, hand_index=hand)
        while not game.is_terminal:
            seat = game.to_act
            obs = game.observation(seat)
            if seat == 0:
                r = client.post("/act", json={"observation": obs.to_dict(), "hand_results": pending},
                                headers=AUTH)
                pending = []
                assert r.status_code == 200, f"/act failed: {r.json()}"
                reply = r.json()
                assert reply.get("hand_end_errors", []) == [], \
                    f"on_hand_end crashed: {reply['hand_end_errors'][0]['traceback']}"
                action = coerce_action(reply["action"])
                assert action in obs.legal_actions, \
                    f"choose_action returned {reply['action']!r}; legal were {[a.name for a in obs.legal_actions]}"
                chosen.append(action)
            else:
                # Random opponent that raises sometimes, so your bot faces bets.
                raises = [a for a in obs.legal_actions if a in (Action.RAISE_HALF, Action.RAISE_POT)]
                action = rng.choice(raises) if raises and rng.random() < raise_rate else Action.CHECK_CALL
            game.apply(action)
        pending.append(game.hand_result(0).to_dict())
    r = client.post("/hand-end", json={"results": pending}, headers=AUTH)  # end of half
    assert r.status_code == 200 and r.json()["hand_end_errors"] == []
    return chosen


# ---------------------------------------------------------------------------
# Your bot plays correctly
# ---------------------------------------------------------------------------

def test_names_are_set(bot):
    _, client = bot
    info = client.get("/").json()
    assert isinstance(info["participant"], str) and info["participant"]
    assert isinstance(info["agent_name"], str) and info["agent_name"]


def test_new_half_works(bot):
    _, client = bot
    r = start_half(client)
    assert r.status_code == 200, f"new_half crashed: {r.json()}"


def test_every_decision_is_legal(bot):
    # 200 hands with a random opponent: every choose_action answer must be legal.
    _, client = bot
    start_half(client)
    play_hands(client, 200)


def test_works_in_both_seats_and_both_halves(bot):
    _, client = bot
    for half, seat in ((1, 0), (2, 1)):
        assert start_half(client, half, seat).status_code == 200
        play_hands(client, 30, seed=f"half{half}")


def test_handles_aggressive_opponent(bot):
    # An opponent that raises whenever it can: your bot must cope with
    # big bets and all-ins.
    _, client = bot
    start_half(client)
    play_hands(client, 100, seed="aggressive", raise_rate=1.0)


def test_memory_is_fresh_each_half(bot):
    module, client = bot
    start_half(client)
    play_hands(client, 10)
    start_half(client, half=2, seat=1)
    after_reset = dict(module.MEMORY)
    fresh = load_agent()
    fresh.new_half(AgentConfig(), fresh.MEMORY)
    assert after_reset.keys() == fresh.MEMORY.keys(), "memory should only hold what new_half() puts there"


# ---------------------------------------------------------------------------
# The server plumbing still works (only fails if you edited it)
# ---------------------------------------------------------------------------

def test_token_is_checked(bot):
    _, client = bot
    body = {"half": 1, "seat": 0, "config": {}}
    assert client.post("/new-half", json=body).status_code == 401
    assert client.post("/new-half", json=body, headers={"X-Agent-Token": "wrong"}).status_code == 401


def test_bad_token_is_refused_before_the_body_is_read(bot):
    # Without the right token the bot must answer 401 without parsing the body,
    # so junk sent by anyone else can't slow your bot down.
    _, client = bot
    junk = b"this is not JSON" * 1000
    assert client.post("/act", content=junk).status_code == 401
    assert client.post("/act", content=junk, headers={"X-Agent-Token": "wrong"}).status_code == 401


def test_malformed_body_is_rejected(bot):
    _, client = bot
    start_half(client)
    assert client.post("/act", content=b"not json", headers=AUTH).status_code == 400
    assert client.post("/act", json=[1, 2, 3], headers=AUTH).status_code == 400


def test_act_before_new_half_is_refused(bot):
    _, client = bot
    obs = Game(make_deal(1, 0), 0).observation(0).to_dict()
    assert client.post("/act", json={"observation": obs, "hand_results": []}, headers=AUTH).status_code == 409


def test_crashes_are_reported_to_the_judge(bot):
    # If choose_action raises, the judge must get a 500 with your traceback.
    module, client = bot

    def broken(obs, memory):
        raise ValueError("oops")

    module.choose_action = broken
    start_half(client)
    obs = Game(make_deal(1, 0), 0).observation(0).to_dict()
    r = client.post("/act", json={"observation": obs, "hand_results": []}, headers=AUTH)
    assert r.status_code == 500
    assert "ValueError: oops" in r.json()["error"] and "Traceback" in r.json()["traceback"]


def test_reads_port_from_environment():
    source = open(os.path.join(BUNDLE, "agent.py"), encoding="utf-8").read()
    assert 'os.environ.get("PORT"' in source, "agent.py must read its port from $PORT"
    assert '0.0.0.0' in source, "agent.py must listen on 0.0.0.0 so the judge can reach it"


# ---------------------------------------------------------------------------
# Add your own tests below, e.g. "with two aces preflop my bot raises".
# ---------------------------------------------------------------------------


def test_params_load_and_scenario_override(bot):
    from bot import strategy
    P = strategy.load_params()
    P["scenarios"] = [{"when": {"street": "PREFLOP"}, "probs": {"CHECK_CALL": 1}}]
    mem = {}
    strategy.new_half(10, mem)
    obs = Game(make_deal(1, 0), 0).observation(0)
    assert all(strategy.decide(obs, mem, P) == Action.CHECK_CALL for _ in range(20))


def test_decisions_fit_time_budget(bot):
    import time
    _, client = bot
    start_half(client)
    t = time.perf_counter()
    n = len(play_hands(client, 100, seed="timing"))
    assert (time.perf_counter() - t) / n < 0.02, "decisions too slow"


def test_shipped_params_are_competition_legal():
    # Training-only overrides live in train/local/ (git-ignored) and must never reach bot/params.json.
    import json
    P = json.load(open(os.path.join(BUNDLE, "bot", "params.json"), encoding="utf-8"))
    # Hard limits: 8 s thinking per half (incl. ~1.5 s+ request overhead we can't see), 1 s per request.
    assert P["time_budget"] <= 5.0, "time_budget must leave margin under the 8 s/half time bank"
    assert P["mc_n"] * P["sample_cost"] < 0.05, "one decision must stay far below the 1 s per-request limit"


def test_vectorised_scoring_matches_the_evaluator():
    import numpy as np
    from bot.equity import omaha_scores
    from evaluator import omaha_score
    r = random.Random(3)
    cards = [r.sample(range(36), 10) for _ in range(200)]
    got = omaha_scores(np.array([c[:5] for c in cards]), np.array([c[5:] for c in cards]))
    assert list(got) == [omaha_score(c[:5], c[5:]) for c in cards]


def test_mc_equity_is_a_probability_on_every_street():
    from bot.equity import mc_equity
    r = random.Random(4)
    for k in (0, 3, 4, 5):
        s = r.sample(range(36), 5 + k)
        assert 0.0 <= mc_equity(s[:5], s[5:], 50, r) <= 1.0


def test_learned_equity_model_is_used_and_a_broken_one_falls_back(bot):
    from bot import strategy
    P = strategy.load_params()
    obs = Game(make_deal(1, 0), 0).observation(0)

    class Good:
        predict = staticmethod(lambda hole, board: 0.42)

    class Broken:
        @staticmethod
        def predict(hole, board):
            raise RuntimeError("bad weights")

    old = strategy.equity_model
    try:
        for model, expected in ((Good, 0.42), (Broken, None)):
            strategy.equity_model = model
            mem = {}
            strategy.new_half(10, mem)
            eq = strategy._equity(obs, mem, P)
            assert (eq == expected) if expected is not None else (0.0 <= eq <= 1.0)
    finally:
        strategy.equity_model = old


def test_stronger_baselines_always_return_a_legal_action():
    import opponents
    r = random.Random(5)
    for name, (act, end) in opponents.STYLES.items():
        mem = {"rng": random.Random(6)}
        for hand in range(40):
            game = Game(make_deal(name, hand), button_seat=hand % 2, hand_index=hand)
            while not game.is_terminal:
                obs = game.observation(game.to_act)
                a = act(obs, mem) if game.to_act == 0 else r.choice(obs.legal_actions)
                assert a in obs.legal_actions
                game.apply(a)
            if end:
                end(game.hand_result(0), mem)
