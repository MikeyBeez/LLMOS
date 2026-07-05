# LLMOS — The Librarian

*How memory is organized, appraised, and retrieved. A design.*

Status: design, 2026-07-04.

## The problem

The LLMOS disk — the brain — is filling with memories: prompts, responses, results, protocols, catalog entries, topic records. So far they go in flat and come back by recall. That works while the store is small and every memory is roughly equal. It breaks at scale, for two reasons. A flat store has no shape, so retrieval either drags in too much or misses the one thing that mattered. And a flat store treats every memory as equal, when they are not — a fact you lean on every day and a throwaway intermediate from a single task deserve to be remembered differently, and, more importantly, *retrieved* differently. The system has to know two things it does not yet know about each memory: where it belongs, and what it is worth. Knowing those, and acting on them, is a librarian's job.

## The librarian is a process, not a schema

The tempting answer is a bigger database schema. That is at most half of it. A library is not its shelving; it is the librarian who decides where a book goes, whether it earns shelf space, when to pull it, and when to move it to the annex. LLMOS already runs this exact kind of process elsewhere — the maintenance pass that consolidates notes, audits their claims, and archives dead artifacts. The librarian is that pattern made a standing part of the operating system: a background daemon that continuously organizes memory. And it runs in idle time on the cheap Mac model, so the expensive processor never spends a cycle shelving books. The classifier is qwen; the librarian is the same cheap tier doing slower, patient housekeeping between your turns.

## Where things live: a hierarchy, not a heap

Memory gets a shape borrowed from a real library's classification. Topics form a tree: a narrow topic has a parent that is broader, and the broad topics climb toward a few roots. Particle physics sits under physics under science; mortgages sit under real estate under finance. A memory is shelved at the most specific topic that fits. Alongside the tree there are cross-links — the *uses* edges already in the index — for the cases a strict tree cannot express, where one topic genuinely depends on another that lives in a different branch (options pricing uses statistics). So the true structure is a tree with a few horizontal wires: mostly hierarchy, with dependencies where hierarchy alone would lie.

The hierarchy is what makes retrieval proportionate. Loading a topic pulls that topic's shelf, inherits a little of the general context from its ancestors, and follows its dependency wires — and nothing else. You get the specific plus the necessary background, not the whole library. This is the structured version of the topic routing already built; the librarian's addition is the parent axis and the discipline of shelving at the right level.

Above topics sits one further level: the project. A project — LLMOS itself, or any other body of work — is the outermost category, the root above the topic trees, and it is a hard scope rather than a soft hint. It is the first cut applied and the coarsest: choose the project, then route within it to a topic, then descend the topic tree. Working in LLMOS, the librarian considers only LLMOS's shelves; another project's memory is not in the room, however much a word might overlap. This is what keeps the LLMOS kernel and a training kernel from some other project — same word, unrelated meaning — from ever landing in the same window.

## What a memory is worth: appraisal and tiers

Every memory carries an importance the librarian assigns and keeps revising. Four tiers are enough to start. *Pinned* memories are the boot ROM — identity, standing configuration, the things that must load by exact key and must never be evicted or forgotten. *Important* memories are the working reference: reliable, frequently useful, eligible to be recalled on a good match. *Trivial* memories are kept but held at arm's length — retrieved only on a strong, specific match, never volunteered. *Ephemeral* memories are session scratch — the intermediates of a task — and are evicted from the window promptly and archived or dropped soon after.

Importance is not declared once; it is earned from signals the librarian can actually measure. How often a memory is recalled, and how recently. Whether the processes that loaded it went on to succeed — a memory that keeps showing up in good outcomes is load-bearing; one that gets loaded and ignored is noise. How many topics depend on it — a memory many things point at is infrastructure. Its provenance — trusted outranks untrusted. And any explicit mark, from you or from a process, which pins it outright. These combine into a score, and the score decays: a memory not used in a long time drifts down through the tiers unless something reinforces it. This is the forgetting curve, on purpose. Nothing important fades, because use keeps reinforcing it; trivia sinks, which is exactly what should happen to trivia.

## When it is worth retrieving: the worthiness gate

This is the part you named directly, and it is the point of the whole design. Retrieval is not free — every memory paged into the window costs tokens, which cost money and slow the next inference. So retrieval has to clear a bar. The bar is worth, and worth is relevance to the current goal multiplied by the memory's importance, judged against the budget for the window. A pinned memory always loads. An important, clearly relevant memory loads. A trivial memory has to be *very* on-point to clear the bar, because its low importance means only high relevance can carry it. Everything below the bar stays on disk.

Retrieval is also tiered, so the system spends its window budget cheaply first. When a topic is loaded, the librarian brings in the pinned and important memories for that topic, its ancestors' key context, and its dependencies — and stops. Only if that does not satisfy the goal does a second page-fault reach deeper, into the trivial shelf, for a specific miss. This mirrors a cache that serves the hot set first and only walks to cold storage on a miss. The result is the answer to "when is it worthwhile": worthwhile is a computed number, not a guess, and the default is restraint — load the hot, important, relevant few; leave the rest shelved until something specifically asks for it.

## What the librarian actually does

Its duties are the verbs of the job. It *shelves*: when a memory is written, it routes the memory to a topic, places it at the right level of the hierarchy, and gives it a starting importance and tier. It *appraises*: it re-scores importance from the access and outcome signals, and applies decay, so the tiers stay honest over time. It *consolidates*: it finds duplicates and near-duplicates and merges them, and it summarizes clusters of trivia into a single compact memory — memory compaction, which is also how the window stays small. It *prunes*: it demotes what has decayed and moves the coldest trivia to an archive tier, never deleting, because a library has an annex, not an incinerator. It *cross-references*: it maintains the parent and dependency edges so the map of what-relates-to-what stays current. And it *catalogs*: it keeps the topic index — the card catalog of prompts, responses, keywords, and links — which is the thing retrieval consults to decide what exists and what it is worth.

## How it fits what already exists

Very little of this is from scratch; most of it is the coping stones on walls already standing. The topic router and the topic index are the card catalog. EVICT is the librarian clearing the reading desk when you are done with a book. The idle scheduler and the cheap Mac model are the hours and the staff — the librarian works between your turns, on the small model, off the critical path. The page-fault recall is the request slip. What the librarian genuinely adds on top is four things: the parent axis that turns flat topics into a hierarchy, the importance score and its tiers, the worthiness gate that decides retrieval by relevance times importance against a budget, and the decay-and-consolidate loop that keeps the whole thing from silting up.

## What to build first

The smallest useful version is not the whole daemon. It is: give each memory a tier and an importance score with a couple of real signals behind it (access count and recency, plus an explicit pin), add a parent field to topics, and put the worthiness gate in front of retrieval so the window loads pinned-and-important-and-relevant first and holds trivia back. That alone makes retrieval proportionate and gives "is this worth pulling" a real answer. The patient parts — consolidation, summarization, archival, outcome-based re-appraisal — are the second pass, and they are precisely the work that wants to run slowly in the background on the cheap model, which is where the librarian is supposed to live anyway.

## The disposition underneath

A librarian is trusted because they are conservative with what they discard and disciplined about what they surface. The LLMOS librarian keeps the same ethos the rest of the system has: it archives rather than deletes, so nothing important is lost to an overconfident prune; every shelving and appraisal is written to the trace, so its judgments are auditable and reversible; and it is bounded, doing a little each idle period rather than a heroic sweep. It is not trying to be clever. It is trying to make sure that when the expensive processor reaches for something, the right thing is there, the trivial things are not in the way, and the reach was worth making.
