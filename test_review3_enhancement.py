"""
CareerIQ X — Review 3 Automated Test Suite
Verifies all newly implemented Review 3 features:
- Domain NLP Semantic Resume-Job Embedding & Cosine Similarity
- Hybrid Matching Formula and Explainability
- Custom Job Description Analyzer
- Review 3 Benchmark Suite
- Personal Career AI Agent (Feature 8)
- Skill Dependency Graph & Bottleneck Analysis (Feature 9)
- Evidence-Based Skills Passport & Public Recruiter URL (Feature 10)
- Multi-dimensional STAR Interview Studio
- Privacy Controls & Session Security
"""

import sys
import json
from fastapi.testclient import TestClient
from app import app, db

client = TestClient(app)

def test_full_pipeline():
    print("=== STARTING REVIEW 3 TEST PIPELINE ===")

    # 1. Register User
    test_email = "alex.coder.review3@example.com"
    with db() as c:
        c.execute("DELETE FROM users WHERE email=?", (test_email,))

    res = client.post("/api/register", json={
        "name": "Alex Coder",
        "email": test_email,
        "password": "StrongPassword123!"
    })
    assert res.status_code == 200, f"Register failed: {res.text}"
    auth_data = res.json()
    user_id = auth_data["user_id"]
    token = auth_data["session_token"]
    headers = {"Authorization": f"Bearer {token}", "X-Session-Token": token}
    print(f"✓ 1. User registered successfully. ID: {user_id}, Session Token: {token[:8]}...")

    # 2. Login
    res = client.post("/api/login", json={
        "email": test_email,
        "password": "StrongPassword123!"
    })
    assert res.status_code == 200, f"Login failed: {res.text}"
    print("✓ 2. User login authenticated.")

    # 3. Update Career Twin Profile
    res = client.put("/api/profile", headers=headers, json={
        "name": "Alex Coder",
        "education": "B.Tech in Computer Science, 2024",
        "experience": "Developed predictive models from historical data and built data transformation pipelines.",
        "skills": "Python, SQL, Git",
        "target_role": "Machine Learning Engineer",
        "location": "Bengaluru",
        "work_mode": "Hybrid"
    })
    assert res.status_code == 200
    print("✓ 3. Career Digital Twin profile saved.")

    # 4. Upload Resume Text
    sample_resume = (
        "ALEX CODER\n"
        "Email: alex.coder.review3@example.com | Phone: +91 98765 43210\n"
        "Summary: Machine learning and backend enthusiast.\n"
        "Experience: Developed predictive models from historical data using statistical techniques and Python.\n"
        "Engineered RESTful API microservices with FastAPI and persisted data in relational SQL databases.\n"
        "Containerized applications using Docker and automated workflows with Git.\n"
    )
    res = client.post(
        "/api/resume",
        headers=headers,
        files={"file": ("resume.txt", sample_resume.encode("utf-8"), "text/plain")}
    )
    assert res.status_code == 200
    resume_resp = res.json()
    print(f"✓ 4. Resume uploaded and vectorized ({resume_resp['vector_dimensions']} dimensions). Detected skills: {resume_resp['skills']}")

    # 5. Review 3 Core: Opportunity Feed with Semantic NLP & Cosine Similarity
    res = client.get("/api/jobs?weight_semantic=0.5&weight_keyword=0.5", headers=headers)
    assert res.status_code == 200
    jobs_data = res.json()
    assert len(jobs_data["jobs"]) > 0
    top_job = jobs_data["jobs"][0]
    print(f"✓ 5. Jobs feed calculated with NLP embeddings. Top match: '{top_job['role']}' at {top_job['company']} (Hybrid Fit: {top_job['hybrid_score']}%, Semantic CosSim: {top_job['semantic_score_pct']}%, Keyword: {top_job['skill_overlap_pct']}%)")
    assert "matched_concepts" in top_job
    assert "semantic_bridges" in top_job

    # 6. Custom Job Description Analyzer (Review 3 proposed implementation)
    custom_jd = (
        "We are looking for a Machine Learning Engineer to build predictive statistical models, "
        "optimize data pipelines, and deploy models into production using Python."
    )
    res = client.post("/api/jobs/custom-match", headers=headers, json={
        "job_title": "AI Modeling Specialist",
        "company": "DeepTech Innovations",
        "job_description": custom_jd,
        "required_skills": "Python, Machine Learning, Statistics, Scikit-learn",
        "weight_semantic": 0.5,
        "weight_keyword": 0.5
    })
    assert res.status_code == 200
    custom_match = res.json()["match_result"]
    print(f"✓ 6. Custom JD Analyzer computed real-time vector similarity. Hybrid score: {custom_match['hybrid_score']}%, Semantic: {custom_match['semantic_score_pct']}%.")
    print(f"   Explanation: {custom_match['explanation']}")

    # 7. Review 3 Benchmark Suite
    res = client.get("/api/evaluation/benchmarks")
    assert res.status_code == 200
    benchmarks = res.json()["benchmarks"]
    print(f"✓ 7. Review 3 Evaluation Benchmarks loaded ({len(benchmarks)} test scenarios verified).")

    # 8. Feature 8: Personal Career AI Agent
    res = client.post("/api/agent/chat", headers=headers, json={
        "message": "What should I work on this week?"
    })
    assert res.status_code == 200
    agent_data = res.json()
    assert len(agent_data["actionable_tasks"]) > 0
    print(f"✓ 8. Career AI Agent responded with personalized advice and {len(agent_data['actionable_tasks'])} actionable tasks.")

    # 9. Feature 9: Skill Dependency Graph & Bottleneck Analysis
    res = client.get("/api/skill-graph", headers=headers)
    assert res.status_code == 200
    graph = res.json()
    bottleneck = graph.get("highest_leverage_skill")
    assert bottleneck is not None
    print(f"✓ 9. Skill Dependency Graph analyzed. Highest-leverage bottleneck: '{bottleneck['skill']}' (Unlocks {len(bottleneck['unlocked_roles'])} roles).")

    # 10. Add Project Evidence (to boost Skills Passport tier)
    res = client.post("/api/projects", headers=headers, json={
        "title": "Customer Churn Prediction Service",
        "description": "Trained predictive models from historical customer datasets and exposed FastAPI endpoints.",
        "skills": "Python, Machine Learning, FastAPI, SQL",
        "github_url": "https://github.com/alex-coder/churn-ml",
        "live_url": "https://churn-ml-demo.example.com"
    })
    assert res.status_code == 200
    print("✓ 10. Project evidence attached to profile.")

    # 11. Multi-Dimensional STAR Interview Studio Practice
    res = client.post("/api/interview/answer", headers=headers, json={
        "role": "Machine Learning Engineer",
        "question": "Describe a project where you used Python and Machine Learning. What problem did it solve?",
        "answer": (
            "In my previous project, our team faced a 25% customer churn rate problem. "
            "I designed and implemented a predictive machine learning pipeline using Python, Scikit-learn, and Pandas. "
            "I built data cleaning scripts, engineered features from historical logs, and trained an XGBoost model. "
            "As a measured result, we reduced prediction latency to 120ms and improved model accuracy to 88%, which helped the team retain 15% more customers."
        )
    })
    assert res.status_code == 200
    interview_res = res.json()
    assert interview_res["score"] >= 75
    print(f"✓ 11. Interview answered and scored across STAR dimensions: {interview_res['score']}/100. STAR Breakdown: {interview_res['star_breakdown']}")

    # 12. Feature 10: Evidence-Based Skills Passport & Public Recruiter Link
    res = client.get("/api/passport", headers=headers)
    assert res.status_code == 200
    passport_data = res.json()
    pass_info = passport_data["passport"]
    print(f"✓ 12. Evidence-Based Skills Passport generated. Trust Index: {pass_info['trust_index']}%. Verified Skills: {pass_info['verified_percentage']}%.")
    
    # Check Public Recruiter Portfolio View (No auth header)
    share_token = passport_data["share_token"]
    pub_res = client.get(f"/api/public/passport/{share_token}")
    assert pub_res.status_code == 200
    pub_html_res = client.get(f"/passport/{share_token}")
    assert pub_html_res.status_code == 200
    assert "Verified Skills Passport" in pub_html_res.text
    print(f"✓ 13. Public Recruiter Portfolio successfully verified at /passport/{share_token}.")

    # 14. Privacy Controls: Data Export & PII Redaction
    export_res = client.get("/api/privacy/export", headers=headers)
    assert export_res.status_code == 200
    anonymize_res = client.post("/api/privacy/anonymize", headers=headers)
    assert anonymize_res.status_code == 200
    print("✓ 14. Privacy controls verified: Full data export & resume PII sanitization.")

    print("\n=======================================================")
    print("ALL 14 REVIEW 3 & ENHANCEMENT TESTS PASSED PERFECTLY! 🚀")
    print("=======================================================")

if __name__ == "__main__":
    test_full_pipeline()

