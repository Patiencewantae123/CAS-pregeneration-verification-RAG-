import hashlib
import json
import os
import threading
import time
from typing import Dict, List, Tuple

from models.document import Document
from models.nli_edge import NLIEdge, NLIRelation

LABEL_TO_RELATION = {
    "contradiction": NLIRelation.CONTRADICTION,
    "entailment": NLIRelation.ENTAILMENT,
    "neutral": NLIRelation.NEUTRAL,
}


class NLIModelVerifier:
    """Pairwise NLI with a pretrained cross-encoder. Drop-in replacement for NLIVerifier.

    Predictions are cached on disk (one JSON line per premise/hypothesis pair) together with the
    measured inference time, so repeated evaluations are cheap and the model cost is still reportable.
    """

    _write_lock = threading.Lock()   # one writer at a time, e.g. several Streamlit sessions sharing a cache file

    def __init__(self, model_name: str = "cross-encoder/nli-deberta-v3-base",
                 cache_path: str = None, batch_size: int = 16, model=None,
                 contradiction_threshold: float = None):
        # `model`: an already-loaded sentence_transformers.CrossEncoder to reuse (e.g. from a Streamlit cache)
        # `contradiction_threshold`: if set, a pair is a contradiction only when its averaged contradiction
        # probability reaches this value; otherwise the argmax of entailment/neutral is used.
        self.contradiction_threshold = contradiction_threshold
        self.model_name = model_name
        self.cache_path = cache_path
        self.batch_size = batch_size
        self._model = model
        self._labels = None
        self._base = None   # set on threshold views, which load the model through the verifier they came from
        if model is not None:
            config = model.model.config
            self._labels = [config.id2label[i].lower() for i in range(len(config.id2label))]
        self.cache: Dict[str, Dict] = {}
        if cache_path and os.path.exists(cache_path):
            with open(cache_path, encoding="utf-8") as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:   # a line cut short by an interrupted write; it is recomputed
                        continue
                    self.cache[rec["key"]] = rec

    def _load(self):
        if self._base is not None:
            self._base._load()
            self._model, self._labels = self._base._model, self._base._labels
        elif self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name, device="cpu")
            self._labels = [self._model.model.config.id2label[i].lower()
                            for i in range(len(self._model.model.config.id2label))]

    def _key(self, premise: str, hypothesis: str) -> str:
        return hashlib.sha1(f"{self.model_name}\x1f{premise}\x1f{hypothesis}".encode("utf-8")).hexdigest()

    def probabilities(self, pairs: List[Tuple[str, str]]) -> List[Dict[str, float]]:
        """Label probabilities for each (premise, hypothesis) pair."""
        keys = [self._key(p, h) for p, h in pairs]
        missing = list(dict.fromkeys(k for k in keys if k not in self.cache))
        if missing:
            self._load()
            todo = {k: pairs[keys.index(k)] for k in missing}
            items = list(todo.items())
            new = []
            for start in range(0, len(items), self.batch_size):
                batch = items[start:start + self.batch_size]
                t = time.perf_counter()
                scores = self._model.predict([pair for _, pair in batch], apply_softmax=True,
                                             batch_size=self.batch_size, show_progress_bar=False)
                per_pair = (time.perf_counter() - t) / len(batch)
                for (k, _), row in zip(batch, scores):
                    rec = {"key": k, "probs": {lab: float(v) for lab, v in zip(self._labels, row)},
                           "seconds": per_pair}
                    self.cache[k] = rec
                    new.append(rec)
            if self.cache_path:
                try:
                    with self._write_lock, open(self.cache_path, "a", encoding="utf-8") as f:
                        f.write("".join(json.dumps(rec) + "\n" for rec in new))
                except OSError:   # read-only location: keep the predictions in memory only
                    pass
        return [self.cache[k]["probs"] for k in keys]

    def inference_seconds(self, pairs: List[Tuple[str, str]]) -> float:
        """Model time measured when these pairs were first computed (0 for pairs never computed)."""
        return sum(self.cache.get(self._key(p, h), {}).get("seconds", 0.0) for p, h in pairs)

    def _symmetric(self, p_ab: Dict[str, float], p_ba: Dict[str, float]) -> Tuple[NLIRelation, float]:
        avg = {lab: (p_ab[lab] + p_ba[lab]) / 2 for lab in p_ab}
        label = max(avg, key=avg.get)
        theta = self.contradiction_threshold
        if label == "contradiction" and theta is not None and avg["contradiction"] < theta:
            label = max(("entailment", "neutral"), key=avg.get)
        return LABEL_TO_RELATION[label], avg[label]

    def with_threshold(self, contradiction_threshold: float) -> "NLIModelVerifier":
        """A view of this verifier that shares its model and prediction cache but uses another threshold."""
        view = NLIModelVerifier.__new__(NLIModelVerifier)
        view.__dict__.update(self.__dict__)
        view.contradiction_threshold = contradiction_threshold
        view._base = self._base or self   # so a model loaded by any view is loaded once, on the base verifier
        return view

    def compare_documents(self, doc_a: Document, doc_b: Document) -> NLIEdge:
        return self.build_pairwise_matrix([doc_a, doc_b])[0]

    def build_pairwise_matrix(self, documents: List[Document]) -> List[NLIEdge]:
        index = [(i, j) for i in range(len(documents)) for j in range(i + 1, len(documents))]
        pairs = []
        for i, j in index:
            pairs.append((documents[i].content, documents[j].content))
            pairs.append((documents[j].content, documents[i].content))
        probs = self.probabilities(pairs)
        edges = []
        for n, (i, j) in enumerate(index):
            relation, confidence = self._symmetric(probs[2 * n], probs[2 * n + 1])
            edges.append(NLIEdge(documents[i].id, documents[j].id, relation, confidence))
        return edges

    @staticmethod
    def document_pairs(documents: List[Document]) -> List[Tuple[str, str]]:
        pairs = []
        for i in range(len(documents)):
            for j in range(i + 1, len(documents)):
                pairs.append((documents[i].content, documents[j].content))
                pairs.append((documents[j].content, documents[i].content))
        return pairs
