from llm_eval import config
from llm_eval.models import Report


class Evaluator:

    def __init__(self, metrics, weights):

        self.metrics = metrics
        self.weights = weights

    # =========================
    # MAIN EVALUATION
    # =========================

    def evaluate(self, response):

        results = {}

        total_score = 0
        total_weight = 0

        for metric in self.metrics:

            metric_name = metric.name()

            score = metric.compute(
                response.response, tokens=response.tokens, latency=response.latency
            )

            results[metric_name] = score

            weight = self.weights.get(metric_name, 1.0)

            total_score += score * weight
            total_weight += weight

        if total_weight > 0:
            final_score = total_score / total_weight
        else:
            final_score = 0.0

        was_passed = final_score >= config.PASS_THRESHOLD

        report = Report(
            prompt=response.prompt,
            response=response.response,
            latency=round(response.latency, 3),
            tokens=response.tokens,
            keyword_score=results.get("keyword_score", 0),
            length_score=results.get("length_score", 0),
            latency_score=results.get("latency_score", 0),
            embedding_semantic_score=results.get("embedding_semantic_score", 0),
            final_score=round(final_score, 3),
            was_passed=was_passed,
        )

        return [report.model_dump()]

    # =========================
    # ERROR HANDLING
    # =========================

    def evaluate_error(self, prompt, error):

        report = Report(
            prompt=prompt,
            response="",
            error=str(error),
            keyword_score=0,
            length_score=0,
            latency_score=0,
            final_score=0,
            was_passed=False,
        )

        return [report.model_dump()]
