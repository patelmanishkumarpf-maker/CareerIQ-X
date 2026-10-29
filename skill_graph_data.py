"""
CareerIQ X — Skill Dependency Graph & Bottleneck Analysis Engine
Implements Feature 9: Skill Dependency Graph.
- Connects roles, skills, prerequisites, and downstream competencies.
- Identifies the most useful missing skill that unlocks multiple career paths.
"""

from typing import List, Dict, Any, Set

SKILL_GRAPH = {
    # Nodes with category, prerequisites, description, and downstream unlocks
    "nodes": {
        # Core Foundations
        "Python": {
            "name": "Python",
            "category": "Programming Languages",
            "level": "Foundation",
            "prerequisites": [],
            "description": "General-purpose language for backend, automation, and data science.",
            "unlocks": ["FastAPI", "Data Structures", "Pandas", "Scikit-learn", "Testing"]
        },
        "JavaScript": {
            "name": "JavaScript",
            "category": "Programming Languages",
            "level": "Foundation",
            "prerequisites": ["HTML", "CSS"],
            "description": "Core dynamic language of the web for client-side and full-stack interactivity.",
            "unlocks": ["React", "Node.js", "Web APIs"]
        },
        "Java": {
            "name": "Java",
            "category": "Programming Languages",
            "level": "Foundation",
            "prerequisites": ["OOP"],
            "description": "Strongly-typed enterprise language for robust, scalable backend systems.",
            "unlocks": ["Spring Boot", "Enterprise Microservices"]
        },
        "SQL": {
            "name": "SQL",
            "category": "Databases & Storage",
            "level": "Foundation",
            "prerequisites": [],
            "description": "Declarative language to query, model, and manipulate relational data.",
            "unlocks": ["REST API", "Database Optimization", "Data Analytics", "Power BI", "ETL Pipelines"]
        },
        "Git": {
            "name": "Git",
            "category": "Developer Tools",
            "level": "Foundation",
            "prerequisites": [],
            "description": "Distributed version control system for tracking changes and collaborating.",
            "unlocks": ["CI/CD", "Team Collaboration", "Open Source"]
        },
        "Linux": {
            "name": "Linux",
            "category": "Systems & Infrastructure",
            "level": "Foundation",
            "prerequisites": [],
            "description": "Operating system command line, permissions, and server administration.",
            "unlocks": ["Docker", "Networking", "Cloud Fundamentals", "SIEM"]
        },
        "HTML": {
            "name": "HTML",
            "category": "Frontend Development",
            "level": "Foundation",
            "prerequisites": [],
            "description": "Semantic markup language structuring web documents and accessibility.",
            "unlocks": ["CSS", "JavaScript"]
        },
        "CSS": {
            "name": "CSS",
            "category": "Frontend Development",
            "level": "Foundation",
            "prerequisites": ["HTML"],
            "description": "Style sheet language for responsive layouts, Flexbox, and CSS Grid.",
            "unlocks": ["JavaScript", "Responsive Design"]
        },
        "Statistics": {
            "name": "Statistics",
            "category": "Data Science Foundations",
            "level": "Foundation",
            "prerequisites": [],
            "description": "Probability distributions, hypothesis testing, and statistical inference.",
            "unlocks": ["Machine Learning", "Data Visualization", "A/B Testing"]
        },
        "OOP": {
            "name": "OOP",
            "category": "Architecture & Paradigms",
            "level": "Foundation",
            "prerequisites": [],
            "description": "Object-oriented design patterns, encapsulation, polymorphism, and abstraction.",
            "unlocks": ["Java", "Spring Boot", "Clean Architecture"]
        },
        
        # Intermediate Skills
        "REST API": {
            "name": "REST API",
            "category": "Backend Engineering",
            "level": "Intermediate",
            "prerequisites": ["Python", "SQL"],
            "description": "HTTP protocol endpoints, JSON data exchange, and status codes.",
            "unlocks": ["FastAPI", "Authentication", "Microservices"]
        },
        "FastAPI": {
            "name": "FastAPI",
            "category": "Backend Engineering",
            "level": "Intermediate",
            "prerequisites": ["Python", "REST API"],
            "description": "High-performance Python web framework with async support and Pydantic validation.",
            "unlocks": ["Microservices", "API Deployment"]
        },
        "Authentication": {
            "name": "Authentication",
            "category": "Backend Engineering",
            "level": "Intermediate",
            "prerequisites": ["REST API"],
            "description": "Secure identity verification, JWT tokens, session hashing, and RBAC.",
            "unlocks": ["Security Fundamentals", "Enterprise Architecture"]
        },
        "Testing": {
            "name": "Testing",
            "category": "Software Quality",
            "level": "Intermediate",
            "prerequisites": ["Python"],
            "description": "Automated unit tests, integration tests, mocking, and Pytest suites.",
            "unlocks": ["CI/CD", "Production Reliability"]
        },
        "Docker": {
            "name": "Docker",
            "category": "DevOps & Cloud",
            "level": "Intermediate",
            "prerequisites": ["Linux"],
            "description": "Container virtualization for consistent application packaging and deployment.",
            "unlocks": ["CI/CD", "Cloud Fundamentals", "Kubernetes"]
        },
        "React": {
            "name": "React",
            "category": "Frontend Development",
            "level": "Intermediate",
            "prerequisites": ["JavaScript", "HTML", "CSS"],
            "description": "Declarative component library for interactive single-page user interfaces.",
            "unlocks": ["Full Stack Developer", "State Management"]
        },
        "Pandas": {
            "name": "Pandas",
            "category": "Data Science & ML",
            "level": "Intermediate",
            "prerequisites": ["Python"],
            "description": "Data analysis and manipulation library for structured DataFrame workflows.",
            "unlocks": ["Machine Learning", "Data Visualization", "NumPy"]
        },
        "NumPy": {
            "name": "NumPy",
            "category": "Data Science & ML",
            "level": "Intermediate",
            "prerequisites": ["Python"],
            "description": "Multi-dimensional array computing and mathematical vector operations.",
            "unlocks": ["Machine Learning", "Deep Learning"]
        },
        "Data Visualization": {
            "name": "Data Visualization",
            "category": "Data Analytics",
            "level": "Intermediate",
            "prerequisites": ["Statistics", "Excel"],
            "description": "Charts, visual analytics, and visual data storytelling.",
            "unlocks": ["Power BI", "Executive Dashboards"]
        },
        "Excel": {
            "name": "Excel",
            "category": "Data Analytics",
            "level": "Intermediate",
            "prerequisites": [],
            "description": "Spreadsheet calculations, pivot tables, lookup formulas, and data cleaning.",
            "unlocks": ["Power BI", "Business Analytics"]
        },
        "Power BI": {
            "name": "Power BI",
            "category": "Data Analytics",
            "level": "Intermediate",
            "prerequisites": ["SQL", "Excel"],
            "description": "Business intelligence dashboards, automated reports, and DAX expressions.",
            "unlocks": ["Data Analyst", "Enterprise BI"]
        },
        "Networking": {
            "name": "Networking",
            "category": "Systems & Security",
            "level": "Intermediate",
            "prerequisites": ["Linux"],
            "description": "TCP/IP, DNS, HTTP/S, subnetting, firewalls, and routing protocols.",
            "unlocks": ["Cybersecurity Analyst", "Cloud Engineer"]
        },

        # Advanced / Specialization Skills
        "Machine Learning": {
            "name": "Machine Learning",
            "category": "Data Science & ML",
            "level": "Advanced",
            "prerequisites": ["Python", "Statistics", "Pandas", "NumPy"],
            "description": "Supervised and unsupervised predictive modeling, classification, and evaluation.",
            "unlocks": ["Scikit-learn", "MLOps", "Deep Learning"]
        },
        "Scikit-learn": {
            "name": "Scikit-learn",
            "category": "Data Science & ML",
            "level": "Advanced",
            "prerequisites": ["Machine Learning", "Python"],
            "description": "Production Python machine learning toolkit for regression, clustering, and tuning.",
            "unlocks": ["Machine Learning Engineer"]
        },
        "CI/CD": {
            "name": "CI/CD",
            "category": "DevOps & Cloud",
            "level": "Advanced",
            "prerequisites": ["Git", "Docker", "Testing"],
            "description": "Continuous Integration & Deployment pipelines with automated validation.",
            "unlocks": ["Cloud Engineer", "Production Automation"]
        },
        "Cloud Fundamentals": {
            "name": "Cloud Fundamentals",
            "category": "DevOps & Cloud",
            "level": "Advanced",
            "prerequisites": ["Linux", "Networking", "Docker"],
            "description": "Cloud hosting, virtual machines, object storage, IAM, and serverless.",
            "unlocks": ["Cloud Engineer", "AWS", "Terraform"]
        },
        "SIEM": {
            "name": "SIEM",
            "category": "Cybersecurity",
            "level": "Advanced",
            "prerequisites": ["Networking", "Linux"],
            "description": "Security Information and Event Management, log correlation, and threat detection.",
            "unlocks": ["Cybersecurity Analyst", "Incident Response"]
        },
        "Incident Response": {
            "name": "Incident Response",
            "category": "Cybersecurity",
            "level": "Advanced",
            "prerequisites": ["SIEM", "Networking"],
            "description": "Investigating, containing, and remediating security breaches and vulnerabilities.",
            "unlocks": ["Cybersecurity Analyst"]
        },
        "Spring Boot": {
            "name": "Spring Boot",
            "category": "Backend Engineering",
            "level": "Advanced",
            "prerequisites": ["Java", "OOP", "SQL"],
            "description": "Enterprise Java framework for building standalone, production-grade microservices.",
            "unlocks": ["Java Developer", "Enterprise Backend"]
        },
        "Data Structures": {
            "name": "Data Structures",
            "category": "Computer Science Core",
            "level": "Foundation",
            "prerequisites": ["Python"],
            "description": "Arrays, linked lists, trees, graphs, sorting, and algorithmic complexity (Big-O).",
            "unlocks": ["Software Engineering", "Technical Interviews"]
        }
    },
    
    # Career Roles and their required skill set
    "roles": {
        "Python Developer": ["Python", "SQL", "Git", "REST API", "FastAPI", "Testing", "Data Structures"],
        "Backend Developer": ["Python", "SQL", "REST API", "FastAPI", "Authentication", "Git", "Testing"],
        "Full Stack Developer": ["HTML", "CSS", "JavaScript", "React", "Python", "SQL", "REST API", "Git"],
        "Data Analyst": ["SQL", "Python", "Excel", "Statistics", "Power BI", "Data Visualization"],
        "Machine Learning Engineer": ["Python", "Statistics", "Machine Learning", "Pandas", "NumPy", "Scikit-learn", "SQL"],
        "Cloud Engineer": ["Linux", "Networking", "Docker", "Cloud Fundamentals", "CI/CD", "Python", "Git"],
        "Cybersecurity Analyst": ["Networking", "Linux", "Python", "SIEM", "Incident Response", "Authentication"],
        "Frontend Developer": ["HTML", "CSS", "JavaScript", "React", "Git", "Testing"],
        "Java Developer": ["Java", "OOP", "SQL", "Spring Boot", "REST API", "Git", "Testing"]
    }
}


def analyze_skill_dependency_graph(user_skills: List[str], target_role: str = "Python Developer") -> Dict[str, Any]:
    """
    Computes graph status, missing prerequisite paths, and identifies
    the single HIGHEST-LEVERAGE missing skill unlocking the most career paths.
    """
    user_skills_set = {s.lower().strip() for s in user_skills}
    nodes_info = []
    missing_skills = set()
    acquired_skills = set()
    
    # Classify each node
    for name, data in SKILL_GRAPH["nodes"].items():
        is_acquired = name.lower() in user_skills_set or any(name.lower() in s for s in user_skills_set)
        if is_acquired:
            acquired_skills.add(name)
        else:
            missing_skills.add(name)
            
        prereqs = data["prerequisites"]
        prereqs_met = all(p.lower() in user_skills_set for p in prereqs)
        
        nodes_info.append({
            "name": name,
            "category": data["category"],
            "level": data["level"],
            "status": "mastered" if is_acquired else ("ready" if prereqs_met else "locked"),
            "prerequisites": prereqs,
            "unlocks": data["unlocks"],
            "description": data["description"]
        })
        
    # Build Edges
    edges = []
    for name, data in SKILL_GRAPH["nodes"].items():
        for unlock in data["unlocks"]:
            if unlock in SKILL_GRAPH["nodes"]:
                edges.append({
                    "source": name,
                    "target": unlock,
                    "type": "prerequisite_of"
                })

    # High-Leverage Missing Skill Bottleneck Analysis:
    # Measures how many roles and downstream skills are unlocked by acquiring each missing skill
    leverage_ranking = []
    for skill in missing_skills:
        skill_data = SKILL_GRAPH["nodes"].get(skill, {})
        downstream = skill_data.get("unlocks", [])
        
        # Count roles that require this skill
        roles_unlocked = [role for role, r_skills in SKILL_GRAPH["roles"].items() if skill in r_skills]
        
        # Calculate leverage score
        score = (len(roles_unlocked) * 2.5) + (len(downstream) * 1.5)
        
        # Target role bonus
        target_role_skills = SKILL_GRAPH["roles"].get(target_role, [])
        if skill in target_role_skills:
            score += 3.0
            
        leverage_ranking.append({
            "skill": skill,
            "category": skill_data.get("category", "General"),
            "level": skill_data.get("level", "Intermediate"),
            "leverage_score": round(score, 1),
            "unlocked_roles": roles_unlocked,
            "unlocked_downstream_skills": downstream,
            "is_target_role_requirement": skill in target_role_skills,
            "prerequisites": skill_data.get("prerequisites", [])
        })
        
    leverage_ranking.sort(key=lambda x: x["leverage_score"], reverse=True)
    
    # Top bottleneck / high-leverage skill
    top_bottleneck = leverage_ranking[0] if leverage_ranking else None
    bottleneck_insight = ""
    if top_bottleneck:
        bottleneck_insight = (
            f"'{top_bottleneck['skill']}' is your #1 highest-leverage missing skill (Leverage Score: {top_bottleneck['leverage_score']}). "
            f"Acquiring it directly advances {len(top_bottleneck['unlocked_roles'])} target career paths "
            f"({', '.join(top_bottleneck['unlocked_roles'][:3])}) and unlocks {len(top_bottleneck['unlocked_downstream_skills'])} downstream competencies "
            f"({', '.join(top_bottleneck['unlocked_downstream_skills'][:3])})."
        )
    else:
        bottleneck_insight = "All fundamental skills in the dependency graph have been acquired! Focus on advanced project proof."

    return {
        "nodes": nodes_info,
        "edges": edges,
        "acquired_count": len(acquired_skills),
        "missing_count": len(missing_skills),
        "total_nodes": len(SKILL_GRAPH["nodes"]),
        "highest_leverage_skill": top_bottleneck,
        "bottleneck_insight": bottleneck_insight,
        "leverage_ranking": leverage_ranking[:6],
        "target_role": target_role
    }

