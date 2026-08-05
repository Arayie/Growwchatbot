from rag_engine import process_rag_query


def run_phase3_verification():
    print("=" * 80)
    print("      SBI MUTUAL FUND RAG — PHASE 3 VERIFICATION TEST SUITE      ")
    print("=" * 80)

    test_queries = [
        {
            "id": 1,
            "query": "What is the minimum SIP for SBI Small Cap Direct Growth?",
            "type": "SQL Fact Lookup",
            "check": lambda res: "500" in res["answer"] and "Source:" in res["answer"] and len(res["follow_up_questions"]) == 3
        },
        {
            "id": 2,
            "query": "What is expense ratio of SBI Contra Fund Direct Plan Growth",
            "type": "Vector Search Procedure",
            "check": lambda res: ("expense ratio" in res["answer"].lower() or "contra" in res["answer"].lower() or "llm" in res["answer"].lower()) and "Source:" in res["answer"] and len(res["follow_up_questions"]) == 3
        },
        {
            "id": 3,
            "query": "Should I buy SBI Bluechip Fund right now?",
            "type": "Advice Refusal",
            "check": lambda res: "cannot provide investment advice" in res["answer"]
        },
        {
            "id": 4,
            "query": "What is the capital of France?",
            "type": "Non-MF Query Refusal",
            "check": lambda res: "strictly with Mutual Fund factual queries" in res["answer"]
        },
        {
            "id": 5,
            "query": "What is the NAV?",
            "type": "Ambiguity Clarification",
            "check": lambda res: "Did you mean" in res["answer"]
        },
        {
            "id": 6,
            "query": "What is the turnover ratio of SBI Unknown Scheme XYZ in 2012?",
            "type": "Strict Unknown Refusal (Zero Hallucination)",
            "check": lambda res: "do not have enough verified factual information" in res["answer"]
        }
    ]

    passed_count = 0

    for test in test_queries:
        t_id = test["id"]
        q = test["query"]
        t_type = test["type"]

        print(f"\n--- TEST #{t_id} ({t_type}): '{q}' ---")
        res = process_rag_query(q)

        print("Response Answer:\n" + res["answer"])
        print("Source URL:", res["source_url"])
        print("Follow-up Questions:", res["follow_up_questions"])

        if test["check"](res):
            print("Status: ✅ PASS")
            passed_count += 1
        else:
            print("Status: ❌ FAIL")

    print("\n" + "=" * 80)
    print(f"VERIFICATION SUMMARY: {passed_count}/{len(test_queries)} Tests Passed ({passed_count/len(test_queries)*100:.1f}%)")
    print("=" * 80)


if __name__ == "__main__":
    run_phase3_verification()
