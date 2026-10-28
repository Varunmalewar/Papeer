"""Viva-style eval for the MOCAA* paper QA. Run: python tests/eval_questions.py

Each question is POSTed to the backend with the MOCAA* paper focused. PASS
requires every required phrase-group to appear (any one variant per group)
and none of the forbidden phrases to appear. Prints a markdown table.
"""
import json
import os
import time
import urllib.request

API = "http://127.0.0.1:8000/chat"
KEY = os.environ["PAPEER_API_KEY"]
USER = os.environ.get("PAPEER_EVAL_USER", "lib-test-user")
TITLE = "springer_paper_bg3[1] (1)"

TESTS = [
    dict(id="a", q=(
        "Quote the abstract sentence about runtime improvement and give the "
        "10,000-node row of Table 2."),
        need=[["35-40", "35–40"], ["23.4", "23.4 ± 2.1", "23.4±2.1"], ["41.2"], ["87.3"], ["1.76"]],
        forbid=[]),
    dict(id="b", q=(
        "The abstract claims 35-40% runtime improvement but Table 2 shows "
        "1.76-2.09x speedup. Which is right?"),
        need=[["35-40", "35–40"], ["1.76"], ["does not explain", "doesn't explain", "not explained"]],
        forbid=["average vs best-case", "averaged", "best case", "best-case"]),
    dict(id="c", q=(
        "Why should we believe the GNN adds value when ablation is future work?"),
        need=[["95.5", "97.0", "98.1", "Table 3"], ["suggestive", "not proven", "not an ablation"],
              ["min", "minimum"]],
        forbid=["compelling evidence", "strong proof"]),
    dict(id="d", q=(
        "Why does h = min(Euclidean, learned) give 43.2% fewer node expansions?"),
        need=[["does not state", "does not explain", "not proven", "cannot", "does not follow"]],
        forbid=["because the learned heuristic is better", "explains the reduction"]),
    dict(id="e", q=(
        "The paper claims optimality, so why is the admissibility rate 96.8% "
        "and optimality gap 0.31%?"),
        need=[["100", "does not state", "does not explain"]],
        forbid=[]),
    dict(id="f", q=(
        "Table 4 shows MOCAA* accuracy of 96.8%. Is the algorithm still optimal?"),
        need=[["100"], ["96.8"], ["conflict", "does not reconcile", "does not explain"]],
        forbid=[]),
    dict(id="g", q="How many inputs does the MLP take?",
        need=[["5", "five"], ["4", "four", "does not resolve"]],
        forbid=[]),
    dict(id="h", q="How many routing objectives does the abstract list?",
        need=[["three", "3"], ["four", "4"]],
        forbid=[]),
    dict(id="i", q="What does the paper say about quantum routing results?",
        need=[["does not state", "does not report", "future work"]],
        forbid=[]),
    dict(id="j", q="What are the limitations of MOCAA*?",
        need=[["500", "resource"], ["obstacle"], ["sparse", "1000"]],
        forbid=[]),
]


def ask(q: str) -> dict:
    body = json.dumps({
        "session_id": f"evalq_{int(time.time() * 1000)}",
        "query": q,
        "doc_scope_ids": [USER],
        "doc_titles": [TITLE],
    }).encode()
    req = urllib.request.Request(API, data=body, headers={
        "Content-Type": "application/json", "X-API-Key": KEY, "X-Papeer-User": USER,
    })
    with urllib.request.urlopen(req, timeout=600) as resp:
        return json.loads(resp.read())


def norm(s: str) -> str:
    return s.lower().replace("–", "-").replace("—", "-").replace(" ", " ")


def grade(ans: str, t: dict) -> tuple[bool, str]:
    low = norm(ans)
    missing = []
    for group in t["need"]:
        if not any(norm(v) in low for v in group):
            missing.append(group[0])
    bad = [f for f in t["forbid"] if norm(f) in low]
    return (not missing and not bad), "; ".join(
        [f"missing: {m}" for m in missing] + [f"forbidden: {b}" for b in bad])


def main():
    rows = []
    for t in TESTS:
        try:
            data = ask(t["q"])
            ans = data.get("answer", "")
            route = data.get("route", "?")
        except Exception as e:
            ans, route = f"ERROR: {e}", "error"
        ok, note = grade(ans, t)
        rows.append((t["id"], route, "PASS" if ok else "FAIL", note))
        print(f"{t['id']} {route:14s} {'PASS' if ok else 'FAIL'} {note}", flush=True)
        time.sleep(2)
    print("\n| Test | Route | Result | Note |\n|---|---|---|---|")
    for r in rows:
        note = (r[3] or "")[:80].replace("|", "/")
        print(f"| {r[0]} | {r[1]} | {r[2]} | {note} |")


if __name__ == "__main__":
    main()
