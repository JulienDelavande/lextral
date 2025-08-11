import math
from typing import List, Dict, Tuple

def _softmax(xs: List[float], tau: float) -> List[float]:
    if len(xs) == 0:
        return []
    mx = max(xs)
    exps = [math.exp((x - mx) / tau) for x in xs]
    s = sum(exps)
    return [e / s for e in exps]

def _normalize(scores: Dict[str, float]) -> Dict[str, float]:
    s = sum(v for v in scores.values() if v > 0)
    if s <= 0:
        return {k: 0.0 for k in scores}
    return {k: v / s for k, v in scores.items()}

def _score_neighbors(
    neighbors: List[Tuple[str, str, float]],  # (label_text, text, sim)
    strategy: str = "weighted",
    power: float = 2.0,
    tau: float = 0.1,
) -> Dict[str, float]:
    """
    Retourne un dict {label: score} non normalisé.
    """
    scores: Dict[str, float] = {}

    if strategy == "majority":
        for label, _, _ in neighbors:
            scores[label] = scores.get(label, 0.0) + 1.0
        return scores

    if strategy == "weighted":
        for label, _, sim in neighbors:
            w = sim ** power
            scores[label] = scores.get(label, 0.0) + w
        return scores

    if strategy == "softmax":
        sims = [sim for (_, _, sim) in neighbors]
        wts = _softmax(sims, tau=tau)
        for (label, _, _), w in zip(neighbors, wts):
            scores[label] = scores.get(label, 0.0) + w
        return scores

    raise ValueError(f"Unknown strategy '{strategy}'")

def _format_evidence(neighbors: List[Tuple[str, str, float]], probs: Dict[str, float], top_n: int = 3) -> str:
    cls_sorted = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:top_n]
    nb_preview = "\n".join([f"- {lbl} (sim={sim:.3f})" for (lbl, _, sim) in neighbors[:top_n]])
    cls_preview = ", ".join([f"{lbl}:{p:.2f}" for lbl, p in cls_sorted])
    return f"probs≈[{cls_preview}]\nneighbors:\n{nb_preview}"
