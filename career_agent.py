"""
CareerIQ X — Personal Career AI Agent Engine
Implements Feature 8: Personal Career AI Agent.
- Understands saved user profile, completed projects, interview scores, and roadmap.
- Answers 'What should I work on this week?' and turns recommendations into actionable tasks.
- Contextual, personalized co-pilot reasoning.
"""

import json
from typing import Dict, Any, List

def run_agent_reasoning(
    prompt: str,
    user_name: str,
    target_role: str,
    skills: List[str],
    matched_skills: List[str],
    missing_skills: List[str],
    projects: List[dict],
    interviews: List[dict],
    applications: List[dict],
    roadmap: List[dict],
    passport_data: dict,
    bottleneck_skill: dict
) -> Dict[str, Any]:
    """
    Intelligent context-aware reasoning engine for the Personal Career AI Agent.
    Synthesizes exact candidate state into personalized, non-generic advice and actionable tasks.
    """
    p_low = prompt.lower().strip()
    
    first_name = user_name.split()[0] if user_name else "there"
    actionable_tasks = []
    response_text = ""
    suggested_followups = []
    
    # 1. Intent: "What should I work on this week?"
    if any(k in p_low for k in ["this week", "what should i work on", "next step", "what to do", "priority"]):
        priority_skill = missing_skills[0] if missing_skills else (bottleneck_skill.get("skill") if bottleneck_skill else "FastAPI")
        second_skill = missing_skills[1] if len(missing_skills) > 1 else "Testing"
        
        response_text = (
            f"Here is your focused, high-leverage weekly plan for **{target_role}**, {first_name}:\n\n"
            f"1. **Core Priority Gap**: Focus on **{priority_skill}**. Based on your dependency graph and target role checklist, "
            f"this is your most critical gap to address.\n"
            f"2. **Portfolio Evidence**: You currently have {len(projects)} project(s). Building a mini-project that showcases "
            f"**{priority_skill}** will elevate your Skills Passport from self-reported to *Project-Evidenced*.\n"
            f"3. **Interview Preparation**: Practice articulating real engineering tradeoffs involving {priority_skill} in the Interview Studio.\n\n"
            f"I have prepared 3 specific actionable tasks you can commit to this week:"
        )
        
        actionable_tasks = [
            {
                "title": f"Study {priority_skill} fundamentals & design patterns",
                "category": "Learning",
                "skill": priority_skill,
                "deadline": "Day 2",
                "details": f"Complete hands-on exercises covering core {priority_skill} concepts and syntax."
            },
            {
                "title": f"Build a verifiable demo project featuring {priority_skill}",
                "category": "Project Evidence",
                "skill": priority_skill,
                "deadline": "Day 5",
                "details": f"Create an open-source repository implementing {priority_skill} with a clear README and test cases."
            },
            {
                "title": f"Complete 1 mock interview question for {target_role}",
                "category": "Interview Studio",
                "skill": target_role,
                "deadline": "Day 7",
                "details": f"Aim for at least 75 points using the Situation-Action-Result framework."
            }
        ]
        
        suggested_followups = [
            f"How do I build a project for {priority_skill}?",
            "Review my interview practice scores",
            "What is my highest-leverage missing skill?"
        ]

    # 2. Intent: High leverage / bottleneck skills
    elif any(k in p_low for k in ["bottleneck", "highest leverage", "unlock", "graph"]):
        b_name = bottleneck_skill.get("skill", "SQL") if bottleneck_skill else "SQL"
        b_roles = bottleneck_skill.get("unlocked_roles", []) if bottleneck_skill else ["Backend Developer", "Data Analyst"]
        b_downstream = bottleneck_skill.get("unlocked_downstream_skills", []) if bottleneck_skill else ["REST API", "Database Optimization"]
        
        response_text = (
            f"According to your Skill Dependency Graph, your **#1 highest-leverage missing skill** is **{b_name}**.\n\n"
            f"**Why this matters:**\n"
            f"- It directly unlocks **{len(b_roles)} major career paths**: {', '.join(b_roles[:4])}.\n"
            f"- It is a prerequisite for **{len(b_downstream)} downstream engineering skills**: {', '.join(b_downstream[:4])}.\n"
            f"- Mastering {b_name} gives you the highest return on your learning time across all potential career transitions."
        )
        
        actionable_tasks = [
            {
                "title": f"Master {b_name} foundational workflows",
                "category": "Skill Building",
                "skill": b_name,
                "deadline": "This week",
                "details": f"Work through practical tutorials and write functional code for {b_name}."
            },
            {
                "title": f"Integrate {b_name} into your active project",
                "category": "Evidence",
                "skill": b_name,
                "deadline": "Next week",
                "details": f"Document your implementation in your Project Portfolio to verify this skill."
            }
        ]
        
        suggested_followups = [
            "What should I work on this week?",
            f"What project can I build using {b_name}?",
            "Check my Skills Passport status"
        ]

    # 3. Intent: Project ideas or evidence verification
    elif any(k in p_low for k in ["project", "proof", "portfolio", "evidence"]):
        target_skill = missing_skills[0] if missing_skills else "FastAPI"
        response_text = (
            f"To build credible proof for **{target_role}**, recruiters look for practical problem-solving rather than generic tutorials.\n\n"
            f"Here is a recommended project architecture to demonstrate **{target_skill}**:\n\n"
            f"**Project Concept**: *Automated Data Pipeline & Service for {target_role}*\n"
            f"- **Problem Solved**: Ingests and processes realistic domain data, exposes RESTful endpoints, and provides measurable output.\n"
            f"- **Skills Demonstrated**: {target_skill}, {', '.join(matched_skills[:2]) if matched_skills else 'Python, Git'}.\n"
            f"- **Proof Artifacts**: GitHub repository with clear setup instructions, automated unit tests, and a live or recorded walkthrough.\n\n"
            f"Once built, add it to your Projects & Proof section to earn a **Project-Evidenced badge** in your Skills Passport!"
        )
        
        actionable_tasks = [
            {
                "title": f"Initialize repository for {target_skill} portfolio project",
                "category": "Projects",
                "skill": target_skill,
                "deadline": "Day 3",
                "details": "Set up Git repo, requirements, and basic architecture skeleton."
            },
            {
                "title": "Add automated tests & test coverage report",
                "category": "Quality",
                "skill": "Testing",
                "deadline": "Day 6",
                "details": "Include at least 5 unit tests proving reliability."
            }
        ]
        
        suggested_followups = [
            "How does this project improve my Skills Passport?",
            "What should I work on this week?",
            "Prepare an interview response for this project"
        ]

    # 4. Intent: Interview preparation & critique
    elif any(k in p_low for k in ["interview", "mock", "question", "star method", "answer"]):
        num_interviews = len(interviews)
        avg_score = round(sum(i.get("score", 0) for i in interviews) / max(1, num_interviews)) if interviews else 0
        
        response_text = (
            f"Here is an evaluation of your interview readiness for **{target_role}**:\n\n"
            f"- **Practice Sessions Logged**: {num_interviews}\n"
            f"- **Average Practice Score**: {avg_score} / 100\n\n"
            f"**Key Coaching Feedback**:\n"
            f"1. **Structure with STAR**: Ensure every answer clearly states the Situation, Task, your personal Action ('I designed...', 'I implemented...'), and the measured Result.\n"
            f"2. **Avoid Passive Voice**: Say what you specifically contributed, not just what 'the team' did.\n"
            f"3. **Quantify Results**: Mention runtime improvements, error reduction, or business value where possible.\n\n"
            f"Ready for a practice round? Head to the Interview Studio to test your response against our multi-dimensional rubric."
        )
        
        actionable_tasks = [
            {
                "title": f"Practice role introduction for {target_role}",
                "category": "Interview",
                "skill": target_role,
                "deadline": "Today",
                "details": "Focus on concise career narrative and technical motivations."
            },
            {
                "title": "Draft a STAR framework story for a challenging bug",
                "category": "Interview",
                "skill": "Problem Solving",
                "deadline": "Tomorrow",
                "details": "Detail the diagnosis, technical resolution, and preventative safeguards."
            }
        ]
        
        suggested_followups = [
            "What should I work on this week?",
            "Review my target role requirements",
            "What is my Skills Passport trust score?"
        ]

    # 5. Intent: Skills Passport & Recruiter credibility
    elif any(k in p_low for k in ["passport", "recruiter", "trust", "verification", "share"]):
        trust_index = passport_data.get("trust_index", 0)
        tier_counts = passport_data.get("tier_counts", {})
        
        response_text = (
            f"Your **Evidence-Based Skills Passport** currently holds a **Trust Index of {trust_index}%** "
            f"({passport_data.get('credibility_status', 'Moderate')}).\n\n"
            f"**Evidence Breakdown**:\n"
            f"- 🟢 **Project-Evidenced (100% Trust)**: {tier_counts.get('Project-Evidenced', 0)} skills\n"
            f"- 🔵 **Interview-Passed (75% Trust)**: {tier_counts.get('Interview-Passed', 0)} skills\n"
            f"- 🟡 **Resume-Detected (50% Trust)**: {tier_counts.get('Resume-Detected', 0)} skills\n"
            f"- ⚪ **Self-Reported (25% Trust)**: {tier_counts.get('Self-Reported', 0)} skills\n\n"
            f"**Recommendation**: Converting your self-reported skills into Project-Evidenced skills will make your "
            f"public shareable recruiter link stand out dramatically over candidates with simple unverified bullet lists."
        )
        
        actionable_tasks = [
            {
                "title": "Link code repository or live demo to your projects",
                "category": "Portfolio",
                "skill": "Verification",
                "deadline": "This week",
                "details": "Ensure every project has verifiable artifacts attached."
            }
        ]
        
        suggested_followups = [
            "What should I work on this week?",
            "What is my highest-leverage missing skill?",
            "How do I share my portfolio with recruiters?"
        ]

    # 6. Default general career guidance
    else:
        response_text = (
            f"I have reviewed your Career Digital Twin for **{target_role}**, {first_name}.\n\n"
            f"- **Target Role**: {target_role}\n"
            f"- **Identified Skills**: {len(skills)} skills ({len(matched_skills)} matching target role)\n"
            f"- **Top Priority Gaps**: {', '.join(missing_skills[:3]) if missing_skills else 'Checklist complete! Focus on project proof.'}\n"
            f"- **Active Projects**: {len(projects)} | **Applications Tracked**: {len(applications)}\n\n"
            f"You can ask me specific questions like:\n"
            f"- *'What should I work on this week?'*\n"
            f"- *'What is my highest-leverage missing skill?'*\n"
            f"- *'What project can I build to prove {missing_skills[0] if missing_skills else 'my skills'}?'*\n"
            f"- *'How can I improve my interview scores?'*"
        )
        suggested_followups = [
            "What should I work on this week?",
            "What is my highest-leverage missing skill?",
            "Evaluate my interview readiness"
        ]

    return {
        "reply": response_text,
        "actionable_tasks": actionable_tasks,
        "suggested_followups": suggested_followups
    }

