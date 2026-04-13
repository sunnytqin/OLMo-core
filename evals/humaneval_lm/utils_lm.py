import re


def process_results(doc, results):
    """Process loglikelihood results for code perplexity.

    For output_type=loglikelihood, results is [(loglikelihood, is_greedy)].
    We compute perplexity over the canonical_solution tokens only
    (the prompt is context, not scored).
    """
    ll, _is_greedy = results[0]
    solution = doc["canonical_solution"]
    _words = len(re.split(r"\s+", solution))
    _bytes = len(solution.encode("utf-8"))
    return {
        "word_perplexity": (ll, _words),
        "byte_perplexity": (ll, _bytes),
        "bits_per_byte": (ll, _bytes),
    }
