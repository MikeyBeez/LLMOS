# pallets/flask — repo knowledge

## PROTOCOL: build the OUTPUT MAP first (always, whether used or not)

Flask is a web framework; its bugs cluster in routing and in the text a
command/handler PRODUCES. Before patching any flask issue, build a symbolic
OUTPUT MAP of whatever structured output is in play — a CLI table, an error
message, a header set, a serialized response. Do it unconditionally; it is
cheap and often it *is* the fix. It is symbolic, not sampled: read the code,
do not guess.

A map is: for each field/column of the output —
  source  : the exact attribute/expression that feeds it (e.g. `rule.subdomain`)
  header  : DERIVED from the source name, not invented — `attr.title()`
            (`rule.subdomain` -> "Subdomain", `rule.host` -> "Host").
            NEVER copy a label from the issue reporter's mock; the reporter's
            example is usually the WRONG format and is a strong distractor.
  mode    : if a flag selects between alternative sources, record BOTH branches
            (flask: `app.url_map.host_matching` selects `rule.host` vs
            `rule.subdomain`; the column header and value both switch).

Then render the output FROM the map. The model decides only *that* a field is
wanted; the header spelling, the mode branches, and the alignment are computed.

### Worked example — `flask routes` domain column (issue 5063)
    row source : app.url_map.iter_rules()  (each is a `rule`)
    mode flag  : app.url_map.host_matching
    columns    : Endpoint  <- rule.endpoint
                 Methods   <- sorted(rule.methods - ignored)
                 [domain]  <- host_matching ? rule.host  : rule.subdomain
                              header        = host_matching ? "Host" : "Subdomain"
                              (present only if any route has a domain)
                 Rule      <- rule.rule
    -> host_matching=False renders "Subdomain"; =True renders "Host".
    Both are required by separate tests; covering only one is the classic miss.

### Why this is the rule
The model produced the reporter's word "Domain" as a single column five times by
sampling — while the two correct words fall out of `attr.title()` with no model
choice at all. Derivable output is DERIVED, not generated (see engineering
pattern: neuro-symbolic split — LLM decides WHAT, scaffold computes HOW spelled).

## Environment
- flask 2.3.x era -> Python 3.8+ (uv fine). Tests: `pytest tests/`.
