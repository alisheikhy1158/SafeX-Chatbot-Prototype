"""
Retrieval evaluation harness for the SafeX FAQ Assistant.

This measures whether the retriever surfaces the *correct* knowledge-base
chunk for a set of hand-labeled test questions.

Metrics reported:
  - Hit Rate @ 1  : top-ranked chunk is the expected one
  - Hit Rate @ 3  : expected chunk appears anywhere in the top 3
  - Mean Reciprocal Rank (MRR): rewards ranking the correct chunk higher

Run:
  python evaluate_retriever.py
"""

from retriever import KnowledgeRetriever

# Each test case pairs a realistic user question with the knowledge-base
# chunk id it SHOULD retrieve (see knowledge_base.py for ids). Written to
# deliberately include paraphrasing / indirect wording, not just keyword
# matches, since that's exactly what embeddings should handle better than
# TF-IDF.
TEST_CASES = [
    {"question": "How can I get in touch with your team?", "expected_id": "contact"},
    {"question": "What's your phone number?", "expected_id": "contact"},
    {"question": "Do you help companies with online security?", "expected_id": "cybersecurity"},
    {"question": "Can you build me a website?", "expected_id": "web-development"},
    {"question": "What is SafeX's social impact program?", "expected_id": "trust"},
    {"question": "Is there a fund that supports students?", "expected_id": "trust"},
    {"question": "What kind of companies do you work with?", "expected_id": "clients"},
    {"question": "Do you offer internships or training programs?", "expected_id": "skill-development-centre"},
    {"question": "What's SafeX's mission?", "expected_id": "mission"},
    {"question": "Can you help with photography or video for my product launch?", "expected_id": "creative-media"},
    {"question": "Do you offer marketing services?", "expected_id": "digital-marketing"},
    {"question": "What countries do you operate in?", "expected_id": "clients"},
    {"question": "Tell me about SafeX as a company.", "expected_id": "about"},
    {"question": "Do you use AI to automate business processes?", "expected_id": "ai-automation"},
    {"question": "Can you manage our cloud infrastructure?", "expected_id": "cloud-solutions"},
]


def evaluate(retriever: KnowledgeRetriever, test_cases=TEST_CASES, top_k=3):
    hits_at_1 = 0
    hits_at_k = 0
    reciprocal_ranks = []
    rows = []

    for case in test_cases:
        results = retriever.retrieve(case["question"], top_k=top_k, min_score=0.0)
        retrieved_ids = [r.id for r in results]

        rank = None
        if case["expected_id"] in retrieved_ids:
            rank = retrieved_ids.index(case["expected_id"]) + 1  # 1-indexed

        if rank == 1:
            hits_at_1 += 1
        if rank is not None:
            hits_at_k += 1
            reciprocal_ranks.append(1.0 / rank)
        else:
            reciprocal_ranks.append(0.0)

        rows.append(
            {
                "question": case["question"],
                "expected": case["expected_id"],
                "retrieved_top": retrieved_ids[0] if retrieved_ids else None,
                "rank_of_expected": rank,
                "top_score": round(results[0].score, 3) if results else None,
            }
        )

    n = len(test_cases)
    metrics = {
        "hit_rate_at_1": hits_at_1 / n,
        f"hit_rate_at_{top_k}": hits_at_k / n,
        "mrr": sum(reciprocal_ranks) / n,
        "n_test_cases": n,
    }
    return metrics, rows


def print_report(metrics, rows):
    print("=" * 70)
    print("SafeX FAQ Assistant — Retriever Evaluation")
    print("=" * 70)
    for row in rows:
        status = "✓" if row["rank_of_expected"] == 1 else ("~" if row["rank_of_expected"] else "✗")
        print(f"[{status}] \"{row['question']}\"")
        print(
            f"      expected={row['expected']}  "
            f"retrieved_top={row['retrieved_top']}  "
            f"rank={row['rank_of_expected']}  "
            f"score={row['top_score']}"
        )
    print("-" * 70)
    print(f"Hit Rate @ 1 : {metrics['hit_rate_at_1']:.1%}")
    print(f"Hit Rate @ 3 : {metrics['hit_rate_at_3']:.1%}")
    print(f"MRR          : {metrics['mrr']:.3f}")
    print(f"Test cases   : {metrics['n_test_cases']}")
    print("=" * 70)


if __name__ == "__main__":
    retriever = KnowledgeRetriever()
    metrics, rows = evaluate(retriever)
    print_report(metrics, rows)
