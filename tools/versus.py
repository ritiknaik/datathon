"""Play two bots against each other exactly like the tournament, and print the verdict.

    python tools/versus.py --a r-poc --b origin/other-branch
    python tools/versus.py --a . --b 3d5fd02 --deals 1000 --seed 7

A and B can each be a git ref (branch, origin/branch, commit) or a folder holding an agent.py.
A ref is checked out into a temporary git worktree, so your working tree is never touched, and
each bot runs from its own folder with its own bot/, params and code. A is "agent1".

Like the tournament: each half is played separately, both bots are RESTARTED between the halves
(fresh servers, so nothing carries over), both halves use the same cards, and the verdict is the
95% confidence interval on A's bb/deal over the paired halves (win 1 / chop 0.5 / loss 0).
Local only: it uses this checkout's judge_engine.py and the ports below.
"""
import argparse
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def prepare(ref, tmp, name):
    """Return a folder containing the bot for this ref (a folder is used as is)."""
    if os.path.isdir(ref):
        return os.path.abspath(ref)
    path = os.path.join(tmp, name)
    subprocess.run(["git", "-C", ROOT, "worktree", "add", "--detach", path, ref], check=True, capture_output=True)
    return path


def start(folder, port):
    env = {**os.environ, "PORT": str(port), "BOT_TRACE": ""}
    return subprocess.Popen([sys.executable, "agent.py"], cwd=folder, env=env,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def judge(*args):
    return subprocess.run([sys.executable, "judge_engine.py", *args], cwd=ROOT, capture_output=True, text=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="bot A (agent1): git ref or folder")
    ap.add_argument("--b", required=True, help="bot B (agent2): git ref or folder")
    ap.add_argument("--deals", type=int, default=1000, help="deals per half (tournament: 1000)")
    ap.add_argument("--seed", default="1", help="match seed; the tournament's is secret")
    ap.add_argument("--ports", nargs=2, type=int, default=[5008, 5009])
    ap.add_argument("--json", help="also save the combined result here")
    ap.add_argument("--keep", help="folder to keep both halves WITH every hand logged (for analysis)")
    a = ap.parse_args()
    pa, pb = a.ports
    tmp = tempfile.mkdtemp(prefix="versus_")
    if a.keep:
        os.makedirs(a.keep, exist_ok=True)
    folders = []
    try:
        fa, fb = prepare(a.a, tmp, "bot_a"), prepare(a.b, tmp, "bot_b")
        folders = [f for f in (fa, fb) if f.startswith(tmp)]
        halves = []
        for half in (1, 2):
            procs = [start(fa, pa), start(fb, pb)]  # fresh servers for every half
            try:
                out = os.path.join(a.keep or tmp, f"half{half}_seed{a.seed}.json")
                r = judge("--agent1", f"http://127.0.0.1:{pa}", "--agent2", f"http://127.0.0.1:{pb}",
                          "--seed", str(a.seed), "--deals", str(a.deals), "--half", str(half), "--json", out,
                          *(["--logs"] if a.keep else []))
                errs = [ln for ln in r.stdout.splitlines() if "errors" in ln and "errors 0" not in ln]
                if r.returncode or errs or not os.path.exists(out):
                    print(r.stdout[-1500:], r.stderr[-1500:])
                    sys.exit("half %d failed (see above)" % half)
                halves.append(out)
                print(f"half {half} done", flush=True)
            finally:
                for p in procs:
                    p.kill()
                    p.wait()
        args = ["combine", *halves] + (["--json", a.json] if a.json else [])
        print(f"\nA (agent1) = {a.a}\nB (agent2) = {a.b}\n")
        print(judge(*args).stdout)
    finally:
        for f in folders:
            subprocess.run(["git", "-C", ROOT, "worktree", "remove", "--force", f], capture_output=True)


if __name__ == "__main__":
    main()
