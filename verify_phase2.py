from guardrails import sanitize_user_input
from router import classify_intent


def run_phase2_verification():
    print("=" * 80)
    print("      SBI MUTUAL FUND RAG — PHASE 2 VERIFICATION TEST SUITE      ")
    print("=" * 80)

    test_cases = [
        {
            "id": 1,
            "prompt": "What is the capital of France?",
            "expected_intent": "NON_MF_QUERY",
            "check_fn": lambda res: res["intent"] == "NON_MF_QUERY" and "strictly with Mutual Fund" in res["response"]
        },
        {
            "id": 2,
            "prompt": "Should I buy SBI Bluechip Fund right now?",
            "expected_intent": "ADVICE_OR_OPINION",
            "check_fn": lambda res: res["intent"] == "ADVICE_OR_OPINION" and "citation_url" in res and "cannot provide investment advice" in res["response"]
        },
        {
            "id": 3,
            "prompt": "My PAN is ABCDE1234F, what is the minimum SIP for SBI Small Cap Direct Growth?",
            "expected_intent": "FACTUAL_QUERY",
            "check_fn": lambda res: res["intent"] == "FACTUAL_QUERY" and "[REDACTED]" in res["sanitized_query"] and "ABCDE1234F" not in res["sanitized_query"]
        },
        {
            "id": 4,
            "prompt": "What is the NAV?",
            "expected_intent": "AMBIGUOUS_QUERY",
            "check_fn": lambda res: res["intent"] == "AMBIGUOUS_QUERY" and "clarification_needed" in res
        }
    ]

    passed_count = 0

    for test in test_cases:
        t_id = test["id"]
        prompt = test["prompt"]
        expected = test["expected_intent"]

        print(f"\n--- TEST #{t_id}: '{prompt}' ---")

        # Step A: Test Sanitization
        sanitized = sanitize_user_input(prompt)
        print(f"  Sanitized Input : '{sanitized}'")

        # Step B: Test Intent Classification
        result = classify_intent(prompt)
        intent = result.get("intent")
        print(f"  Detected Intent : {intent}")
        print(f"  Result Output   : {result}")

        # Step C: Validate Check Function
        if test["check_fn"](result):
            print(f"  Status          : ✅ PASS (Expected: {expected})")
            passed_count += 1
        else:
            print(f"  Status          : ❌ FAIL (Expected: {expected})")

    print("\n" + "=" * 80)
    print(f"VERIFICATION SUMMARY: {passed_count}/{len(test_cases)} Tests Passed ({passed_count/len(test_cases)*100:.1f}%)")
    print("=" * 80)


if __name__ == "__main__":
    run_phase2_verification()
