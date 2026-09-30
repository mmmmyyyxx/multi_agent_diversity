"""Offline assertions substituted for upstream resource download calls only."""
import importlib.util


def require_spacy_model(name: str) -> None:
    if importlib.util.find_spec(name) is None:
        raise RuntimeError("IFBENCH_LOCAL_SPACY_RESOURCE_MISSING")


def require_nltk_resource(name: str) -> None:
    import nltk
    resources = {"punkt": "tokenizers/punkt", "punkt_tab": "tokenizers/punkt_tab",
        "stopwords": "corpora/stopwords", "averaged_perceptron_tagger_eng": "taggers/averaged_perceptron_tagger_eng"}
    if name not in resources:
        raise RuntimeError("IFBENCH_LOCAL_NLTK_RESOURCE_UNREGISTERED")
    try:
        nltk.data.find(resources[name])
    except LookupError as exc:
        raise RuntimeError("IFBENCH_LOCAL_NLTK_RESOURCE_MISSING") from exc
