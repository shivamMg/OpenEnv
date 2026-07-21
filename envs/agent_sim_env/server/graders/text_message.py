"""ROUGE-L grading for agent text messages."""

from rouge_score import rouge_scorer


class TextMessageGrader:
    """Compute stemmed ROUGE-L F-measure for an assistant text response."""

    def __init__(self) -> None:
        self._scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    def grade(self, actual: str, expected: str) -> float:
        """Return a ROUGE-L score, treating absent expert text as unconstrained."""
        if not expected:
            return 1.0
        return round(self._scorer.score(expected, actual)["rougeL"].fmeasure, 2)
