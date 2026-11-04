"""
CAS v2: the selection layer redesigned from the diagnosis in the manuscript (Section 6).

  v1 problem (measured)                                         v2 change
  1. Near-copies 'entail' each other, so agreement raised        No agreement term.
     the poison's trust.
  2. Provenance, a 0.25 weight, could not outweigh relevance.    Provenance is a gate (>= 0.5) when metadata exist.
  3. The NLI graph caught < 10% of conflicts at 24-40 s per      No NLI graph. A lexical check finds near-copies that
     question, and its notes did not change answers.             disagree, and both sides are quarantined when
                                                                 provenance cannot settle the dispute.
  4. Dropping one supporting passage cost far more than          A passage whose title is named in the question or in
     keeping a distractor.                                       a kept passage is kept too.

Everything is lexical: no model is run, so selection takes milliseconds.
"""
import re

from run_evaluation import DEFAULT_TAU, bm25_scores, tokens

KAPPA = 0.8          # near-copy: this share of the shorter passage's content words also occurs in the other
MIN_WORDS = 8        # passages with fewer content words are never treated as near-copies
GATE = 0.5           # provenance below this is untrusted
W_REL, W_PROV, W_RANK = 0.35, 0.25, 0.20   # the v1 weights, without the agreement term


def divergent_pairs(passages, ids, kappa=KAPPA):
    """Pairs of near-copies in which each side has content words the other lacks: a disputed copy."""
    words = {i: set(tokens(passages[i])) for i in ids}
    out = []
    ids = sorted(ids)
    for a in range(len(ids)):
        for b in range(a + 1, len(ids)):
            i, j = ids[a], ids[b]
            small = min(len(words[i]), len(words[j]))
            if small < MIN_WORDS:
                continue
            shared = len(words[i] & words[j])
            if shared / small >= kappa and words[i] - words[j] and words[j] - words[i]:
                out.append((i, j))
    return out


def _named(title, text):
    t = re.sub(r"\s*\([^)]*\)$", "", title or "").strip()
    return len(t) >= 4 and re.search(r"(?<!\w)" + re.escape(t.lower()) + r"(?!\w)", text.lower()) is not None


def select(question, passages, titles, prov=None, tau=DEFAULT_TAU, kappa=KAPPA, gate=True, quarantine=True, link=True):
    """Return (selected ids, info). `prov` is None when no source metadata exist; `titles` may hold None.

    The switches turn single changes off, for ablations; with all three off this is v1 without the conflict graph.
    """
    n = len(passages)
    pool = set(range(n))
    gated = set()
    if prov is not None and gate:
        gated = {i for i in pool if prov[i] < GATE}
        pool -= gated
    disputes = divergent_pairs(passages, pool, kappa) if quarantine else []
    quarantined = {i for pair in disputes for i in pair}
    pool -= quarantined

    raw = bm25_scores(question, passages)
    hi = max(raw)
    order = sorted(range(n), key=lambda i: -raw[i])
    rank = {i: r for r, i in enumerate(order)}
    p = prov if prov is not None else [0.5] * n
    trust = {i: W_REL * (raw[i] / hi if hi > 0 else 0.5) + W_PROV * p[i] + W_RANK * (1 - rank[i] / n) for i in pool}
    kept = {i for i in pool if trust[i] >= tau}

    linked = set()
    if link and kept:
        context = question + " " + " ".join(passages[i] for i in kept)
        linked = {i for i in pool - kept if _named(titles[i], context)}
    return kept | linked, {"gated": sorted(gated), "quarantined": sorted(quarantined), "linked": sorted(linked),
                           "disputes": disputes}
