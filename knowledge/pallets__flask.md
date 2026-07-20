# pallets/flask -- repo knowledge

## PROTOCOL: build the OUTPUT MAP first (always, whether used or not)

Flask is a web framework; its bugs cluster in routing and in the text a
command/handler PRODUCES. Before patching any flask issue, build a symbolic
OUTPUT MAP of whatever structured output is in play -- a CLI table, an error
message, a header set, a serialized response. Do it unconditionally; it is
cheap and often it IS the fix. It is symbolic, not sampled: read the code, do
not guess.

A map is: for each field of the output --
  source  : the exact attribute/expression that feeds it (read it from the code)
  header  : DERIVED from the source name, not invented -- `attr.title()`.
            NEVER copy a label from the issue reporter's mock; the reporter's
            example is frequently the WRONG format and is a strong distractor.
  mode    : if a config flag selects between ALTERNATIVE source fields for the
            same slot, record every branch -- each branch has its own
            field-derived label, and a fix covering only one branch is the
            classic miss.

Then render the output FROM the map. The model decides only WHICH fields are
wanted; the header spelling, the mode branches and the alignment are computed.

Rationale: derivable output must be DERIVED, not generated. A label is a
function of the field it displays, so it should never be sampled.

## Environment
- flask 2.3.x era -> Python 3.8+ (uv fine). Tests: `pytest tests/`.
