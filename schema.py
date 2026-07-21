"""Canonical shapes for LLMOS data. One place. Everything else references it.

Mikey: "the way to handle this sort of stuff is to build a canonical space
like /etc."

Every analysis error today was a bare string key that failed SILENTLY:

    inst.get("patch")          -> field is gold_patch  -> "0 patches found"
    event.get("result")        -> field is ok          -> "0 of 1039 peeked"
    outcome.get("probe_status")-> written after save   -> "the probe is dead"

`.get()` on a schema you did not verify converts a typo into confident wrong
data. Declaring the shape once turns the same typo into an immediate error
naming the field you probably meant.

USE:
    from schema import INSTANCE, EVENT, OUTCOME, FIX_STATE, sget

    sget(inst, "gold_patch")        # fine
    sget(inst, "patch")             # KeyError: not a field of instance.
                                    #   Did you mean 'gold_patch'? Fields: ...
    INSTANCE.GOLD_PATCH             # "gold_patch" -- typo is an AttributeError
    INSTANCE.validate(inst)         # {"missing": [...], "unexpected": [...]}
"""
import difflib


class Shape:
    def __init__(self, name, required=(), optional=()):
        self.name = name
        self.required = tuple(required)
        self.optional = tuple(optional)
        for f in self.required + self.optional:
            setattr(self, f.upper().replace(".", "_"), f)

    @property
    def fields(self):
        return set(self.required) | set(self.optional)

    def check_key(self, key):
        if key in self.fields:
            return
        near = difflib.get_close_matches(key, sorted(self.fields), n=2, cutoff=0.5)
        raise KeyError(
            "%r is not a field of %s.%s Fields: %s"
            % (key, self.name,
               (" Did you mean %s?" % " or ".join(repr(n) for n in near)) if near else "",
               ", ".join(sorted(self.fields))))

    def get(self, d, key, default=None):
        self.check_key(key)
        return (d or {}).get(key, default)

    def validate(self, d):
        keys = set(d or {})
        return {"missing": sorted(set(self.required) - keys),
                "unexpected": sorted(keys - self.fields)}


# ---- the shapes, as observed on disk 2026-07-20 --------------------------

INSTANCE = Shape(
    "instance",
    required=("instance_id", "repo", "base_commit", "problem_statement",
              "test_patch", "gold_patch", "FAIL_TO_PASS", "PASS_TO_PASS"))

TRACE = Shape(
    "trace",
    required=("phase1", "phase1_meta", "phase2", "phase2_meta",
              "state", "fix_state", "outcome"),
    optional=("phase1_events", "phase2_events", "remedies"))

EVENT = Shape("event", required=("tool", "args", "ok", "error"))

OUTCOME = Shape(
    "outcome",
    required=("id", "resolved", "secs", "patch_bytes",
              "phase1_reason", "phase2_reason"),
    optional=("env_ok", "env_kind", "env_vars", "python", "installs",
              "score_tail", "fix_verified_by_model", "seen_red", "repro_green",
              "probe_green", "probe_status", "given_tests_ok", "given_tests_n",
              "given_tests_regressed", "syntax_ok", "note", "error",
              "attempt", "attempt_secs", "attempts_made", "iteration"))

FIX_STATE = Shape(
    "fix_state",
    required=("seen_red", "repro_green", "fix_verified", "patch_history"),
    optional=("repro_script", "repro_mode", "repro_locked", "probe_script",
              "probe_green", "baseline_pass", "regressions", "checks_run",
              "chain_mechanism", "chain_change", "triage_goal", "triage_repro",
              "triage_unverified", "failed_anchors", "must_observe",
              "patch_attempts", "same_verify_count", "last_verify_sig",
              "rejected_repro_streak", "submitted", "net_retries"))

SHAPES = {s.name: s for s in (INSTANCE, TRACE, EVENT, OUTCOME, FIX_STATE)}


def sget(d, key, default=None, shape=None):
    """Strict get: the key must be declared somewhere in the canonical space."""
    if shape is not None:
        return SHAPES[shape].get(d, key, default) if isinstance(shape, str) \
            else shape.get(d, key, default)
    for s in SHAPES.values():
        if key in s.fields:
            return (d or {}).get(key, default)
    near = []
    for s in SHAPES.values():
        near += difflib.get_close_matches(key, sorted(s.fields), n=1, cutoff=0.5)
    raise KeyError("%r is not a field of any known shape.%s"
                   % (key, (" Did you mean %s?" % " or ".join(repr(n) for n in near))
                      if near else ""))
