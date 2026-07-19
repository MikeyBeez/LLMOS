#!/usr/bin/env python3
"""format_job: try different formats -- ranked by LEGITIMATE signals only.

Mikey's idea (2026-07-19): "create a little job that tries different formats
and passes the question through with the different formats."

The leakage boundary, stated up front: on the benchmark, candidate formats
must NEVER be tried against the hidden tests -- each pass/fail verdict is a
leaked bit, and iterating on it is answer-key optimization. In deployment
(your own CI) that loop is legal. Here, the job ENUMERATES candidates and
RANKS them by evidence the agent legitimately holds:

  signal A (strongest): the attribute the code actually reads. If the diff
    prints a value obtained from rule.subdomain, the label candidate
    "Subdomain" gets the attribute bonus. (The flask lesson.)
  signal B: the repo's EXISTING public tests at base commit -- how do they
    assert on output of this same command/area? Words already asserted-on by
    the maintainers are words the maintainers test with.
  signal C (weakest): words from the issue text. Reporters describe; the
    codebase names.

usage: format_job.py <repo_dir> <patch_file>
Reports each user-facing string literal added by the patch alongside ranked
alternatives. Wire-in target: the worksheet (offer before generation) and the
diff-lint (challenge after).
"""
import os
import re
import sys


def attribute_reads(patch_text):
    """Attributes read in added lines: the code's own vocabulary, weighted."""
    reads = {}
    for ln in patch_text.splitlines():
        if not ln.startswith("+"):
            continue
        for m in re.finditer(r"\b\w+\.([a-z_][a-z0-9_]*)\b", ln):
            a = m.group(1)
            if a not in ("format", "join", "append", "get", "items", "startswith"):
                reads[a] = reads.get(a, 0) + 1
    return reads


def added_labels(patch_text):
    """User-facing string literals introduced by the patch (headers, labels)."""
    labels = []
    for ln in patch_text.splitlines():
        if not ln.startswith("+"):
            continue
        for m in re.finditer(r"[\"\']([A-Z][A-Za-z ]{2,20})[\"\']", ln):
            labels.append(m.group(1))
    return sorted(set(labels))


def existing_test_words(repo_dir):
    """Words the repo's PUBLIC tests (base commit) already assert on in CLI
    output -- maintainer testing vocabulary. Legitimate: these files are part
    of the checkout the agent works in."""
    # LEAKAGE GUARD: read test files from the BASE COMMIT (git show HEAD:),
    # never the working tree -- after scoring, the tree contains the applied
    # hidden test patch, and scanning it reads the answer key. Caught live
    # 2026-07-19: Subdomain/Host appeared in signal B only because the demo
    # ran post-score. Too-good signals are leakage until proven otherwise.
    import subprocess
    words = {}
    ls = subprocess.run(["git", "-C", repo_dir, "ls-files"],
                        capture_output=True, text=True).stdout.splitlines()
    for rel in ls:
        fn = os.path.basename(rel)
        if not (fn.startswith("test") and fn.endswith(".py")):
            continue
        txt = subprocess.run(["git", "-C", repo_dir, "show", "HEAD:" + rel],
                             capture_output=True, text=True).stdout
        if True:
            for m in re.finditer(
                    r"assert\s+[\"\']([A-Z][A-Za-z ]{2,20})[\"\']\s+in", txt):
                w = m.group(1)
                words[w] = words.get(w, 0) + 1
    return words


def rank(repo_dir, patch_file):
    patch = open(patch_file, encoding="utf-8", errors="ignore").read()
    reads = attribute_reads(patch)
    labels = added_labels(patch)
    testwords = existing_test_words(repo_dir)
    print("attributes read by the patch: %s" %
          sorted(reads, key=reads.get, reverse=True)[:8])
    print("maintainer test-asserted words: %s" %
          sorted(testwords, key=testwords.get, reverse=True)[:8])
    print()
    for lab in labels:
        cands = {}
        # candidate from each read attribute (title-cased), scored
        for a, n in reads.items():
            cand = a.replace("_", " ").title().replace(" ", "")
            cands[cand] = cands.get(cand, 0) + 3 * n          # signal A
        for w, n in testwords.items():
            cands[w] = cands.get(w, 0) + 2 * n                # signal B
        cands[lab] = cands.get(lab, 0) + 1                    # signal C (itself)
        ranked = sorted(cands.items(), key=lambda kv: -kv[1])[:5]
        verdict = "OK" if ranked and ranked[0][0] == lab else "SUSPECT"
        print("label %-14r %-8s better-ranked: %s"
              % (lab, verdict,
                 [c for c, _ in ranked if c != lab][:3]))


if __name__ == "__main__":
    rank(sys.argv[1], sys.argv[2])
