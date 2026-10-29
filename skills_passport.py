"""
CareerIQ X — Evidence-Based Skills Passport
Implements Feature 10: Evidence-Based Skills Passport.
- Verifiable trust tiers for each skill (Self-Reported, Resume-Detected, Interview-Passed, Project-Evidenced).
- Computes overall Passport Verification & Trust Index.
- Generates public shareable portfolio URL for recruiters.
"""

import secrets
import json
from datetime import datetime
from typing import List, Dict, Any

TIER_SCORES = {
    "Project-Evidenced": 100,
    "Interview-Passed": 75,
    "Resume-Detected": 50,
    "Self-Reported": 25
}

def generate_share_token():
    """Generates a secure URL-safe share token for recruiter portfolio links."""
    return secrets.token_urlsafe(16)

def calculate_skills_passport(user_profile: dict, projects: list, interview_results: list) -> dict:
    """
    Evaluates every candidate skill against concrete proof sources:
    - Profile text
    - Resume extracted text
    - Project descriptions and tagged skills
    - Interview answers and passing rubric scores
    
    Returns tiered verification status and Passport Trust Index.
    """
    profile_skills_str = user_profile.get("skills") or ""
    resume_text = (user_profile.get("resume_text") or "").lower()
    
    # Parse explicit skills
    raw_skills = [s.strip() for s in profile_skills_str.split(",") if s.strip()]
    
    # Analyze project proof
    project_skills = set()
    project_proof_map = {}
    for p in projects:
        p_title = p.get("title", "")
        p_desc = p.get("description", "")
        p_skills = [s.strip() for s in p.get("skills", "").split(",") if s.strip()]
        for s in p_skills:
            s_low = s.lower()
            project_skills.add(s_low)
            if s_low not in project_proof_map:
                project_proof_map[s_low] = []
            project_proof_map[s_low].append({
                "project_title": p_title,
                "description": p_desc[:120] + ("..." if len(p_desc) > 120 else "")
            })
            
    # Analyze interview proof (passing score >= 60%)
    interview_passed_skills = set()
    interview_proof_map = {}
    for i in interview_results:
        score = i.get("score", 0)
        q = i.get("question", "")
        role = i.get("role", "")
        if score >= 60:
            for s in raw_skills:
                if s.lower() in q.lower() or s.lower() in (i.get("answer") or "").lower():
                    interview_passed_skills.add(s.lower())
                    interview_proof_map[s.lower()] = {
                        "question": q,
                        "score": score,
                        "role": role
                    }

    passport_skills = []
    total_trust_points = 0
    
    for skill in sorted(set(raw_skills)):
        s_low = skill.lower()
        tier = "Self-Reported"
        evidence_detail = "Claimed in personal profile."
        
        # Check Project Evidence (Highest Tier)
        if s_low in project_skills:
            tier = "Project-Evidenced"
            proofs = project_proof_map.get(s_low, [])
            p_names = ", ".join(p["project_title"] for p in proofs[:2])
            evidence_detail = f"Verified with practical project evidence: '{p_names}'."
        # Check Interview Evidence
        elif s_low in interview_passed_skills:
            tier = "Interview-Passed"
            i_info = interview_proof_map[s_low]
            evidence_detail = f"Validated through technical interview practice ({i_info['score']}/100 pts)."
        # Check Resume Evidence
        elif s_low in resume_text:
            tier = "Resume-Detected"
            evidence_detail = "Extracted and identified from uploaded resume document."
            
        trust_weight = TIER_SCORES[tier]
        total_trust_points += trust_weight
        
        passport_skills.append({
            "skill": skill,
            "tier": tier,
            "trust_score": trust_weight,
            "badge_color": "green" if tier == "Project-Evidenced" else "blue" if tier == "Interview-Passed" else "amber" if tier == "Resume-Detected" else "muted",
            "evidence_detail": evidence_detail
        })
        
    num_skills = max(1, len(passport_skills))
    trust_index = round(total_trust_points / num_skills) if passport_skills else 0
    
    # Summary metrics
    tier_counts = {
        "Project-Evidenced": sum(1 for s in passport_skills if s["tier"] == "Project-Evidenced"),
        "Interview-Passed": sum(1 for s in passport_skills if s["tier"] == "Interview-Passed"),
        "Resume-Detected": sum(1 for s in passport_skills if s["tier"] == "Resume-Detected"),
        "Self-Reported": sum(1 for s in passport_skills if s["tier"] == "Self-Reported")
    }
    
    # Credibility statement
    if trust_index >= 75:
        credibility_status = "High Evidence Credibility"
        advice = "Your skills are solidly backed by working projects and interview evaluations."
    elif trust_index >= 50:
        credibility_status = "Moderate Evidence Credibility"
        advice = "Turn your resume-detected skills into project-evidenced portfolio items to strengthen recruiter trust."
    else:
        credibility_status = "Emerging Profile"
        advice = "Add project evidence and complete mock interviews to convert self-reported claims into verified proof."

    return {
        "passport_skills": passport_skills,
        "trust_index": trust_index,
        "credibility_status": credibility_status,
        "advice": advice,
        "tier_counts": tier_counts,
        "total_skills": len(passport_skills),
        "verified_percentage": round(100 * (tier_counts["Project-Evidenced"] + tier_counts["Interview-Passed"]) / max(1, len(passport_skills)), 1)
    }

