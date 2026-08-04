from fastapi.testclient import TestClient
from app import app

client = TestClient(app)


def run_phase4_verification():
    print("=" * 80)
    print("      SBI MUTUAL FUND RAG — PHASE 4 VERIFICATION TEST SUITE      ")
    print("=" * 80)

    # Test 1: GET /api/health
    print("\n--- TEST #1: GET /api/health ---")
    response = client.get("/api/health")
    print("HTTP Status Code:", response.status_code)
    print("Response JSON   :", response.json())
    assert response.status_code == 200
    assert response.json()["status"] == "online"
    print("Status: ✅ PASS")

    # Test 2: POST /api/chat - Factual Query
    print("\n--- TEST #2: POST /api/chat (Factual Query) ---")
    payload2 = {"message": "What is the minimum SIP for SBI Small Cap Direct Growth?"}
    res2 = client.post("/api/chat", json=payload2)
    data2 = res2.json()
    print("HTTP Status Code:", res2.status_code)
    print("Response JSON   :", data2)
    assert res2.status_code == 200
    assert data2["status"] == "success"
    assert data2["intent"] == "FACTUAL_QUERY"
    assert len(data2["follow_up_questions"]) == 3
    print("Status: ✅ PASS")

    # Test 3: POST /api/chat - Advice Refusal
    print("\n--- TEST #3: POST /api/chat (Advice Refusal) ---")
    payload3 = {"message": "Should I buy SBI Bluechip?"}
    res3 = client.post("/api/chat", json=payload3)
    data3 = res3.json()
    print("HTTP Status Code:", res3.status_code)
    print("Response JSON   :", data3)
    assert res3.status_code == 200
    assert data3["intent"] == "ADVICE_OR_OPINION"
    assert "cannot provide investment advice" in data3["answer"]
    print("Status: ✅ PASS")

    # Test 4: POST /api/chat - Non-MF Refusal
    print("\n--- TEST #4: POST /api/chat (Non-MF Refusal) ---")
    payload4 = {"message": "What is the weather today?"}
    res4 = client.post("/api/chat", json=payload4)
    data4 = res4.json()
    print("HTTP Status Code:", res4.status_code)
    print("Response JSON   :", data4)
    assert res4.status_code == 200
    assert data4["intent"] == "NON_MF_QUERY"
    assert "strictly with Mutual Fund" in data4["answer"]
    print("Status: ✅ PASS")

    print("\n" + "=" * 80)
    print("VERIFICATION SUMMARY: 4/4 Tests Passed (100.0%)")
    print("=" * 80)


if __name__ == "__main__":
    run_phase4_verification()
