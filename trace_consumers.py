#!/usr/bin/env python3
"""trace_consumers.py — downstream consumers of SWE-bench agent traces.

A trace is written once (swe_agent_v2._save_trace) but has THREE consumers:

  1. Humans debugging       -> events_from_messages(): compact per-turn event
                               records (tool, args, ok, error signature) so a
                               run can be scanned without parsing raw chat.
                               Stored in the trace as phase1_events/phase2_events.

  2. The agent at runtime   -> the REMEDY STORE (~/swe/remedies.json).
                               After every run, extract_remedies() mines the
                               transcript for (error signature -> fix) pairs
                               via llm_call; remedies_for(repo) injects prior
                               fixes into the next bootstrap goal.
                               BENEFIT: "correct once, never again." The v4
                               astropy run spent ~15 turns discovering that
                               setuptools>=64,<69 satisfies BOTH dep_util and
                               build_editable — the next astropy-family
                               instance should get that lesson at turn 0.
                               Remedies compound across the Lite suite because
                               many instances share repos and error families.

  3. The model at training  -> ~/swe/training/{bootstrap,fix}.jsonl.
                               Successful transcripts saved as ollama chat-
                               format JSONL (system + full tool-call dialog).
                               BENEFIT: traces become fine-tuning data for
                               ornith in its NATIVE tool-call format, with no
                               conversion step. Also usable immediately as
                               retrieved few-shot exemplars in the system
                               prompt before any fine-tune happens.

All entry points are exception-safe by contract of the caller (harvest_trace
is wrapped in try/except in swe_agent_v2) — a consumer failure must never
kill or corrupt a run.
"""
import json, os, re, time

from repo_bootstrap_tools import llm_call

REMEDIES = os.path.expanduser("~/swe/remedies.json")
TRAINING_DIR = os.path.expanduser("~/swe/training")


# ---------- consumer 1: events array --------------------------------------

_SIG_PATTERNS = [
    r"ModuleNotFoundError: No module named '[^']+'",
    r"ImportError: cannot import name '[^']+' from [^\n]{0,80}",
    r"AttributeError: [^\n]{0,120}",
    r"error: [^\n]{0,140}",
    r"ERROR: [^\n]{0,140}",
    r"FAILED [^\n]{0,140}",
]


def error_signature(text):
    """Normalize an error blob to a short, matchable one-line signature."""
    text = str(text)
    for pat in _SIG_PATTERNS:
        m = re.search(pat, text)
        if m:
            return m.group(0)[:160]
    line = next((l.strip() for l in text.splitlines() if l.strip()), "")
    return line[:160]


def events_from_messages(messages):
    """Compact per-turn events: pair each assistant tool_call with the
    tool result that follows it. Answers 'what happened' without reading
    the raw transcript."""
    events = []
    for i, m in enumerate(messages):
        if m.get("role") != "assistant" or not m.get("tool_calls"):
            continue
        fn = (m["tool_calls"][0] or {}).get("function", {})
        ev = {"tool": fn.get("name"), "args": fn.get("arguments"),
              "ok": None, "error": None}
        if i + 1 < len(messages) and messages[i + 1].get("role") == "tool":
            raw = messages[i + 1].get("content") or ""
            try:
                res = json.loads(raw)
            except Exception:
                res = {"_raw": raw[:200]}
            if isinstance(res, dict):
                ev["ok"] = res.get("ok", "error" not in res)
                err = res.get("error") or (res.get("stderr")
                                           if res.get("ok") is False else None)
                if err:
                    ev["error"] = error_signature(err)
        events.append(ev)
    return events


def _events_digest(events, max_chars=6000):
    lines = []
    for n, ev in enumerate(events):
        args = json.dumps(ev.get("args"), default=str)[:100]
        s = f"[{n}] {ev['tool']}({args}) ok={ev['ok']}"
        if ev.get("error"):
            s += f"  err={ev['error']}"
        lines.append(s)
    return "\n".join(lines)[-max_chars:]


# ---------- mid-run critic: review the trace, search the error -------------

def critic_review(messages, last_n=12):
    """Detached review of the last N turns. Returns one-paragraph advice
    (or "" if all is well). Uses the model OUTSIDE the agent's context —
    the reviewer sees the digest, not the agent's rationalizations — and
    web-searches the most recent error signature for outside knowledge."""
    from repo_bootstrap_tools import llm_call, _ddg_search
    events = events_from_messages(messages)
    if not events:
        return ""
    recent = events[-last_n:]
    digest = _events_digest(recent, max_chars=4000)
    last_err = next((e["error"] for e in reversed(recent) if e.get("error")),
                    None)
    web_blob = ""
    if last_err:
        hits = _ddg_search(last_err[:120], 3)
        if hits:
            web_blob = "\nWeb results for the latest error:\n" + "\n".join(
                f"- {h['title']}: {h['snippet'][:200]}" for h in hits)
    advice = llm_call(
        system=("You are a detached reviewer watching an agent set up a "
                "repository. Be blunt and concrete. If the agent is doing "
                "fine, reply exactly OK."),
        prompt=(f"Recent turns (tool ok=True/False, err=signature):\n{digest}\n"
                f"{web_blob}\n\n"
                "Is the agent looping, repeating a failed action, destroying "
                "its own progress, or chasing the wrong cause? Reply with ONE "
                "short paragraph of corrective advice naming the exact next "
                "action, or exactly OK if progress is sound."),
        max_tokens=600)
    advice = (advice or "").strip()
    if advice.upper().startswith("OK") and len(advice) < 8:
        return ""
    return advice[:700]


# ---------- consumer 2: remedy store ---------------------------------------

_EXTRACT_SYS = (
    "You extract durable engineering lessons from agent logs. "
    "Respond ONLY with JSON.")

_EXTRACT_PROMPT = """Below is a turn-by-turn digest of an automated agent \
setting up the repository {repo} for testing. Each line is one tool call \
with its outcome.

Identify DURABLE REMEDIES: cases where the agent hit a specific error and a \
specific later action fixed it (evidenced by a later ok=True on the thing \
that had failed). Ignore one-off flakes and anything not clearly resolved.

Respond with JSON exactly like:
{{"remedies": [{{"error_signature": "<short matchable error line>",
                "remedy": "<the concrete action that fixed it, with exact \
package names / version specs / flags>",
                "evidence": "turn N failed, turn M succeeded"}}]}}
Empty list if none.

DIGEST:
{digest}
"""


def extract_remedies(messages, repo, instance_id):
    """Mine one transcript for (error signature -> remedy) pairs via llm_call."""
    events = events_from_messages(messages)
    if not any(ev.get("error") for ev in events):
        return []
    raw = llm_call(_EXTRACT_PROMPT.format(repo=repo,
                                          digest=_events_digest(events)),
                   system=_EXTRACT_SYS, format_json=True,
                   temperature=0.2, max_tokens=1600)
    try:
        got = json.loads(raw)
        items = got.get("remedies", []) if isinstance(got, dict) else []
    except Exception:
        return []
    out, now = [], time.strftime("%Y-%m-%d")
    for it in items:
        sig = str(it.get("error_signature", "")).strip()
        rem = str(it.get("remedy", "")).strip()
        if sig and rem:
            out.append({"error_signature": sig[:200], "remedy": rem[:500],
                        "evidence": str(it.get("evidence", ""))[:120],
                        "repo": repo, "source_instance": instance_id,
                        "date": now})
    return out


def _load_remedies():
    try:
        return json.load(open(REMEDIES))
    except Exception:
        return []


def merge_remedies(new):
    """Dedupe by (repo, normalized signature); append genuinely new ones.
    Returns number added."""
    store = _load_remedies()
    seen = {(r["repo"], r["error_signature"].lower().strip()) for r in store}
    added = 0
    for r in new:
        key = (r["repo"], r["error_signature"].lower().strip())
        if key not in seen:
            store.append(r)
            seen.add(key)
            added += 1
    if added:
        tmp = REMEDIES + ".tmp"
        json.dump(store, open(tmp, "w"), indent=1)
        os.replace(tmp, REMEDIES)
    return added


def remedies_for(repo):
    """Remedies recorded for this repo (exact repo match)."""
    return [r for r in _load_remedies() if r.get("repo") == repo]


def format_remedy_context(remedies, limit=8):
    """Render remedies as a block to append to the phase-1 goal prompt."""
    lines = ["KNOWN REMEDIES from previous runs on this repository — apply "
             "these proactively instead of rediscovering them:"]
    for r in remedies[:limit]:
        lines.append(f"- if you hit: {r['error_signature']}\n"
                     f"  then: {r['remedy']}")
    return "\n".join(lines)


# ---------- per-repo build playbooks (remedies graduate to protocols) ------

PLAYBOOKS = os.path.expanduser("~/swe/playbooks.json")


def _load_playbooks():
    try:
        return json.load(open(PLAYBOOKS))
    except Exception:
        return {}


def playbook_for(repo):
    return _load_playbooks().get(repo)


def synthesize_playbook(repo, bootstrap_events, instance_id):
    """After a successful bootstrap, distill remedies + the winning event
    sequence into ONE procedure. Validated by construction: the sequence
    it summarizes just passed the env gate."""
    from repo_bootstrap_tools import llm_call
    ok_steps = [f"{e['tool']}({json.dumps(e.get('args'), default=str)[:140]})"
                for e in bootstrap_events if e.get("ok")]
    rems = "\n".join(f"- {r['error_signature']}: {r['remedy'][:200]}"
                     for r in remedies_for(repo)[:10])
    raw = llm_call(
        system=("You write concise, exact build playbooks for repositories. "
                "JSON only."),
        prompt=(f"An agent just SUCCESSFULLY set up {repo} for testing. "
                f"Winning tool sequence:\n" + "\n".join(ok_steps[-25:]) +
                f"\n\nKnown remedies for this repo:\n{rems}\n\n"
                "Write the definitive playbook as JSON:\n"
                '  python_version, backend, build_deps (ordered, with pins), '
                'install (the editable-install call), env_vars, '
                'smoke_test (exact working run_smoke_test args), '
                'gotchas (list of one-line warnings)'),
        max_tokens=1600, format_json=True)
    from repo_bootstrap_tools import _extract_json
    pb = _extract_json(raw) or {}
    if not pb:
        return None
    books = _load_playbooks()
    prev = books.get(repo) or {}
    pb_rec = {"playbook": pb,
              "validated_runs": int(prev.get("validated_runs", 0)) + 1,
              "source_instance": instance_id,
              "updated": time.strftime("%Y-%m-%d")}
    books[repo] = pb_rec
    tmp = PLAYBOOKS + ".tmp"
    json.dump(books, open(tmp, "w"), indent=1)
    os.replace(tmp, PLAYBOOKS)
    return pb_rec


def format_playbook_context(rec):
    return ("VALIDATED BUILD PLAYBOOK for this repository (succeeded "
            f"{rec['validated_runs']}x, updated {rec['updated']}) — follow "
            "it directly instead of rediscovering:\n"
            + json.dumps(rec["playbook"], indent=1, default=str)[:2200])


# ---------- consumer 3: training export ------------------------------------

def export_training(messages, inst, tag, resolved):
    """Append one JSONL line in ollama chat format (system + full tool-call
    dialog) to ~/swe/training/<tag>.jsonl. Re-runs of the same instance
    replace the earlier line, so the file holds the latest transcript per
    instance."""
    os.makedirs(TRAINING_DIR, exist_ok=True)
    path = os.path.join(TRAINING_DIR, f"{tag}.jsonl")
    rec = {"instance_id": inst["instance_id"], "repo": inst["repo"],
           "tag": tag, "resolved": bool(resolved),
           "date": time.strftime("%Y-%m-%d"),
           "messages": messages}
    kept = []
    if os.path.exists(path):
        for line in open(path):
            try:
                old = json.loads(line)
                if old.get("instance_id") != inst["instance_id"]:
                    kept.append(line.rstrip("\n"))
            except Exception:
                continue
    kept.append(json.dumps(rec, default=str))
    tmp = path + ".tmp"
    open(tmp, "w").write("\n".join(kept) + "\n")
    os.replace(tmp, path)
    return path


# ---------- partial-trace harvesting ----------------------------------------

def harvest_partials(traces_dir=os.path.expanduser("~/swe/traces_v2"),
                     instances_path=os.path.expanduser("~/swe/instances.json")):
    """Mine interrupted runs' checkpoints (*.partial.json) into the remedy
    store, then archive them as .harvested. Run this BEFORE every rerun —
    the killed instance's trace is usually the one that shows why we
    stopped."""
    try:
        insts = {i["instance_id"]: i for i in json.load(open(instances_path))}
    except Exception:
        insts = {}
    report = []
    import glob as _glob
    for pp in _glob.glob(os.path.join(traces_dir, "*.partial.json")):
        iid = os.path.basename(pp).replace(".partial.json", "")
        repo = insts.get(iid, {}).get("repo") or iid.split("__")[0]
        try:
            blob = json.load(open(pp))
            msgs = blob.get("phase1") or []
            rems = extract_remedies(msgs, repo, iid + " (interrupted)")
            added = merge_remedies(rems)
            report.append({"instance": iid, "turns": len(
                events_from_messages(msgs)), "remedies_new": added})
            os.replace(pp, pp + ".harvested")
        except Exception as e:
            report.append({"instance": iid, "error": str(e)})
    return report


# ---------- orchestration ---------------------------------------------------

def harvest_trace(inst, blob):
    """Run all consumers over a finished trace blob. Mutates blob in place
    (adds phaseN_events + remedies) and returns a summary dict for the log."""
    summary = {}
    for phase in ("phase1", "phase2"):
        if phase in blob:
            blob[phase + "_events"] = events_from_messages(blob[phase])
    # consumer 2: remedy store (mine every run — failures often contain the
    # hardest-won lessons; v4's setuptools range came from a failed run)
    rems = extract_remedies(blob.get("phase1", []), inst["repo"],
                            inst["instance_id"])
    blob["remedies"] = rems
    summary["remedies_extracted"] = len(rems)
    summary["remedies_new"] = merge_remedies(rems)
    # consumer 3: training export (successes only — don't train on flailing)
    out = blob.get("outcome", {})
    if out.get("env_ok"):
        try:
            pb = synthesize_playbook(inst["repo"],
                                     blob.get("phase1_events") or [],
                                     inst["instance_id"])
            summary["playbook"] = bool(pb)
        except Exception:
            summary["playbook"] = "failed"
        export_training(blob["phase1"], inst, "bootstrap",
                        resolved=out.get("resolved"))
        summary["training"] = ["bootstrap"]
        if out.get("resolved") and "phase2" in blob:
            export_training(blob["phase2"], inst, "fix", resolved=True)
            summary["training"].append("fix")
    return summary
