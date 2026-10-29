"""
CareerIQ X — Career Operating System (Review 3 Enhanced Edition)
Includes:
- Review 3 Core: Semantic Resume–Job Matching using Domain NLP Embeddings & Cosine Similarity
- Feature 8: Personal Career AI Agent co-pilot
- Feature 9: Skill Dependency Graph & Bottleneck Analysis
- Feature 10: Evidence-Based Skills Passport with Public Recruiter Portfolio
- Enhanced Opportunity Discovery with Custom Job Description Analyzer & Evaluation Benchmarks
- Multi-dimensional STAR Interview Studio
- Strengthened Authentication, Session Tokens, and Privacy Controls
"""

import os, re, sqlite3, json, hashlib, secrets
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request, Depends
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from passlib.context import CryptContext

import nlp_engine
import skill_graph_data
import skills_passport
import career_agent

APP_DIR = Path(__file__).parent
DB_PATH = APP_DIR / "careeriq.db"
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

app = FastAPI(title="CareerIQ X — Review 3 Enhanced", version="2.0.0")

def db():
    con = sqlite3.connect(DB_PATH, timeout=15.0)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys=ON")
    con.execute("PRAGMA journal_mode=WAL")
    return con

def init_db():
    with db() as c:
        # Base tables
        c.executescript("""
        CREATE TABLE IF NOT EXISTS users(
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS sessions(
            token TEXT PRIMARY KEY,
            user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            expires_at TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS profiles(
            user_id INTEGER PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
            education TEXT DEFAULT '',
            experience TEXT DEFAULT '',
            skills TEXT DEFAULT '',
            target_role TEXT DEFAULT 'Python Developer',
            location TEXT DEFAULT 'India',
            work_mode TEXT DEFAULT 'Any',
            resume_text TEXT DEFAULT '',
            public_share_token TEXT DEFAULT '',
            is_public INTEGER DEFAULT 1,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS projects(
            id INTEGER PRIMARY KEY,
            user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            title TEXT NOT NULL,
            description TEXT DEFAULT '',
            skills TEXT DEFAULT '',
            live_url TEXT DEFAULT '',
            github_url TEXT DEFAULT '',
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS applications(
            id INTEGER PRIMARY KEY,
            user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            company TEXT NOT NULL,
            role TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'Saved',
            url TEXT DEFAULT '',
            notes TEXT DEFAULT '',
            match_score INTEGER DEFAULT 0,
            semantic_score INTEGER DEFAULT 0,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS interview_results(
            id INTEGER PRIMARY KEY,
            user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            role TEXT NOT NULL,
            question TEXT NOT NULL,
            answer TEXT DEFAULT '',
            score INTEGER DEFAULT 0,
            feedback TEXT DEFAULT '',
            star_scores TEXT DEFAULT '{}',
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS agent_messages(
            id INTEGER PRIMARY KEY,
            user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            sender TEXT NOT NULL,
            message TEXT NOT NULL,
            actionable_tasks TEXT DEFAULT '[]',
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS roadmap_tasks(
            id INTEGER PRIMARY KEY,
            user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            title TEXT NOT NULL,
            skill TEXT DEFAULT '',
            category TEXT DEFAULT 'Learning',
            status TEXT NOT NULL DEFAULT 'Pending',
            created_at TEXT NOT NULL
        );
        """)
        
        # Check migrations for older profiles and projects
        try:
            cols = [r["name"] for r in c.execute("PRAGMA table_info(profiles)").fetchall()]
            if "public_share_token" not in cols:
                c.execute("ALTER TABLE profiles ADD COLUMN public_share_token TEXT DEFAULT ''")
            if "is_public" not in cols:
                c.execute("ALTER TABLE profiles ADD COLUMN is_public INTEGER DEFAULT 1")
        except Exception:
            pass

        try:
            cols = [r["name"] for r in c.execute("PRAGMA table_info(projects)").fetchall()]
            if "live_url" not in cols:
                c.execute("ALTER TABLE projects ADD COLUMN live_url TEXT DEFAULT ''")
            if "github_url" not in cols:
                c.execute("ALTER TABLE projects ADD COLUMN github_url TEXT DEFAULT ''")
        except Exception:
            pass

        try:
            cols = [r["name"] for r in c.execute("PRAGMA table_info(interview_results)").fetchall()]
            if "star_scores" not in cols:
                c.execute("ALTER TABLE interview_results ADD COLUMN star_scores TEXT DEFAULT '{}'")
        except Exception:
            pass

init_db()

# Knowledge base
ROLE_SKILLS = skill_graph_data.SKILL_GRAPH["roles"]
ROLE_DESCRIPTIONS = {
    "Python Developer": "Build Python applications, APIs, automation workflows, and backend services.",
    "Full Stack Developer": "Build user-facing web applications, responsive user interfaces, and backend microservices.",
    "Data Analyst": "Turn structured data into actionable insights, executive KPI dashboards, and business recommendations.",
    "Machine Learning Engineer": "Prepare data, engineer features, train predictive models, and deploy production ML systems.",
    "Backend Developer": "Design secure RESTful APIs, business logic, relational databases, and server-side systems.",
    "Frontend Developer": "Build accessible, responsive, and performant web interfaces using React and modern CSS.",
    "Cybersecurity Analyst": "Monitor network systems, analyze SIEM security events, and mitigate cyber vulnerabilities.",
    "Cloud Engineer": "Deploy, automate, and operate reliable applications on Linux and cloud infrastructure with Docker and CI/CD.",
    "Java Developer": "Build object-oriented enterprise applications and microservices using Java and Spring Boot."
}

SYNONYMS = {
    "js": "JavaScript", "javascript": "JavaScript", "py": "Python", "postgres": "SQL", "postgresql": "SQL",
    "mysql": "SQL", "restful api": "REST API", "api development": "REST API", "machine-learning": "Machine Learning",
    "ml": "Machine Learning", "scikit learn": "Scikit-learn", "powerbi": "Power BI", "object oriented programming": "OOP",
    "spring": "Spring Boot", "springboot": "Spring Boot", "git/github": "Git", "data structures and algorithms": "Data Structures",
    "ci-cd": "CI/CD", "github actions": "CI/CD", "docker container": "Docker", "k8s": "Docker"
}

ALL_SKILLS = sorted(set(s for vals in ROLE_SKILLS.values() for s in vals) | {
    "Django", "Flask", "Node.js", "AWS", "Azure", "Communication", "Problem Solving", "Agile", "MongoDB",
    "PostgreSQL", "PyTorch", "TensorFlow", "Pandas", "NumPy", "Excel", "Power BI", "Data Visualization",
    "Statistics", "Linux", "Networking", "Docker", "CI/CD", "Authentication", "Testing", "Accessibility",
    "OOP", "Java", "C++", "C", "Firebase", "Flutter", "Scikit-learn", "SIEM", "Incident Response", "FastAPI"
})

def extract_skills(text: str) -> List[str]:
    low = (text or "").lower()
    found = []
    for skill in ALL_SKILLS:
        pattern = r"(?<![a-z0-9])" + re.escape(skill.lower()) + r"(?![a-z0-9])"
        if re.search(pattern, low):
            found.append(skill)
    for alias, canonical in SYNONYMS.items():
        if alias in low and canonical not in found:
            found.append(canonical)
    return sorted(set(found))

# Session Management
def create_session(user_id: int, conn=None) -> str:
    token = secrets.token_hex(32)
    expires_at = (datetime.utcnow() + timedelta(days=30)).isoformat()
    now = datetime.utcnow().isoformat()
    if conn is not None:
        conn.execute("INSERT INTO sessions(token, user_id, expires_at, created_at) VALUES(?,?,?,?)",
                     (token, user_id, expires_at, now))
    else:
        with db() as c:
            c.execute("INSERT INTO sessions(token, user_id, expires_at, created_at) VALUES(?,?,?,?)",
                      (token, user_id, expires_at, now))
    return token

def user_from_request(request: Request) -> int:
    # 1. Bearer Token
    auth_header = request.headers.get("Authorization")
    token = None
    if auth_header and auth_header.startswith("Bearer "):
        token = auth_header[7:].strip()
    
    # 2. X-Session-Token
    if not token:
        token = request.headers.get("X-Session-Token")
        
    if token:
        with db() as c:
            row = c.execute("SELECT user_id, expires_at FROM sessions WHERE token=?", (token,)).fetchone()
            if row:
                if datetime.fromisoformat(row["expires_at"]) > datetime.utcnow():
                    return row["user_id"]
                else:
                    c.execute("DELETE FROM sessions WHERE token=?", (token,))

    # 3. Fallback for existing local clients: X-User-Id
    user_header = request.headers.get("X-User-Id") or request.query_params.get("user_id")
    if user_header:
        try:
            uid = int(user_header)
            with db() as c:
                if c.execute("SELECT id FROM users WHERE id=?", (uid,)).fetchone():
                    return uid
        except Exception:
            pass

    raise HTTPException(401, "Session expired or missing. Please sign in.")

# Pydantic In models
class AuthIn(BaseModel):
    name: Optional[str] = ""
    email: str
    password: str

class ProfileIn(BaseModel):
    name: Optional[str] = ""
    education: str = ""
    experience: str = ""
    skills: str = ""
    target_role: str = "Python Developer"
    location: str = "India"
    work_mode: str = "Any"

class ProjectIn(BaseModel):
    title: str
    description: str = ""
    skills: str = ""
    live_url: Optional[str] = ""
    github_url: Optional[str] = ""

class ApplicationIn(BaseModel):
    company: str
    role: str
    status: str = "Saved"
    url: str = ""
    notes: str = ""

class ApplicationUpdate(BaseModel):
    company: str
    role: str
    status: str
    url: str = ""
    notes: str = ""

class InterviewIn(BaseModel):
    role: str
    question: str
    answer: str = ""

class CustomMatchIn(BaseModel):
    job_title: str
    company: Optional[str] = "Target Company"
    job_description: str
    required_skills: Optional[str] = ""
    weight_semantic: Optional[float] = 0.5
    weight_keyword: Optional[float] = 0.5

class AgentChatIn(BaseModel):
    message: str

class TaskIn(BaseModel):
    title: str
    skill: Optional[str] = ""
    category: Optional[str] = "Learning"

# Rich Job Opportunities Feed
JOB_FEED = [
    {
        "id": "job-1",
        "company": "Cognizant AI Labs",
        "role": "Machine Learning Engineer",
        "category": "Machine Learning Engineer",
        "location": "Bengaluru / Hybrid",
        "work_mode": "Hybrid",
        "salary_range": "₹12–18 LPA",
        "description": "Seeking an ML Engineer to design predictive models from historical corporate data, build automated feature pipelines, apply Scikit-learn, and deploy scalable ML prediction microservices.",
        "required_skills": ["Python", "Machine Learning", "Pandas", "NumPy", "Scikit-learn", "Statistics", "SQL"],
        "url": "https://www.linkedin.com/jobs/"
    },
    {
        "id": "job-2",
        "company": "Razorpay Payments",
        "role": "Backend Software Engineer",
        "category": "Backend Developer",
        "location": "Bengaluru / Remote",
        "work_mode": "Remote",
        "salary_range": "₹14–22 LPA",
        "description": "Build high-throughput RESTful API endpoints and transactional services in Python and FastAPI. Manage PostgreSQL relational databases, write Pytest test suites, and enforce robust JWT authentication.",
        "required_skills": ["Python", "FastAPI", "SQL", "REST API", "Authentication", "Testing", "Git"],
        "url": "https://www.linkedin.com/jobs/"
    },
    {
        "id": "job-3",
        "company": "Swiggy Tech",
        "role": "Python Developer — Core Services",
        "category": "Python Developer",
        "location": "Hyderabad / Hybrid",
        "work_mode": "Hybrid",
        "salary_range": "₹10–16 LPA",
        "description": "Design asynchronous backend workflows and REST API microservices. Work with complex SQL data queries, data structures, Git version control, and automated continuous delivery.",
        "required_skills": ["Python", "SQL", "Git", "REST API", "FastAPI", "Testing", "Data Structures"],
        "url": "https://www.naukri.com/"
    },
    {
        "id": "job-4",
        "company": "Freshworks Inc",
        "role": "Full Stack Developer",
        "category": "Full Stack Developer",
        "location": "Chennai / On-site",
        "work_mode": "On-site",
        "salary_range": "₹11–17 LPA",
        "description": "Build modern responsive web user interfaces using React, JavaScript, and HTML5/CSS3. Connect with Python and Node REST APIs, design database schemas, and ensure cross-browser responsiveness.",
        "required_skills": ["HTML", "CSS", "JavaScript", "React", "Python", "SQL", "REST API", "Git"],
        "url": "https://www.linkedin.com/jobs/"
    },
    {
        "id": "job-5",
        "company": "Deloitte Analytics",
        "role": "Data Analyst — BI & Strategy",
        "category": "Data Analyst",
        "location": "Mumbai / Hybrid",
        "work_mode": "Hybrid",
        "salary_range": "₹8–14 LPA",
        "description": "Transform transactional data into executive KPI dashboards using Power BI and advanced SQL queries. Clean messy datasets in Excel and Python, conduct statistical correlation analysis, and deliver business recommendations.",
        "required_skills": ["SQL", "Python", "Excel", "Statistics", "Power BI", "Data Visualization"],
        "url": "https://www.naukri.com/"
    },
    {
        "id": "job-6",
        "company": "Zomato Infrastructure",
        "role": "Cloud & DevOps Associate",
        "category": "Cloud Engineer",
        "location": "Gurugram / Hybrid",
        "work_mode": "Hybrid",
        "salary_range": "₹12–19 LPA",
        "description": "Automate deployment workflows with CI/CD and GitHub Actions. Containerize services with Docker, manage Linux servers, configure cloud monitoring, and provision cloud resources.",
        "required_skills": ["Linux", "Networking", "Docker", "Cloud Fundamentals", "CI/CD", "Python", "Git"],
        "url": "https://www.linkedin.com/jobs/"
    },
    {
        "id": "job-7",
        "company": "Infosys Cyber Defense",
        "role": "Cybersecurity Analyst",
        "category": "Cybersecurity Analyst",
        "location": "Pune / On-site",
        "work_mode": "On-site",
        "salary_range": "₹9–15 LPA",
        "description": "Monitor real-time network logs, investigate SIEM alerts, identify intrusions, conduct vulnerability assessments, and implement incident response containment procedures.",
        "required_skills": ["Networking", "Linux", "Python", "SIEM", "Incident Response", "Authentication"],
        "url": "https://www.naukri.com/"
    },
    {
        "id": "job-8",
        "company": "Tata Consultancy Services",
        "role": "Java Enterprise Backend Developer",
        "category": "Java Developer",
        "location": "Kolkata / Hybrid",
        "work_mode": "Hybrid",
        "salary_range": "₹9–15 LPA",
        "description": "Develop scalable enterprise services using Java 17, Spring Boot, Hibernate ORM, and relational SQL. Write JUnit automated tests and build RESTful microservice architectures.",
        "required_skills": ["Java", "OOP", "SQL", "Spring Boot", "REST API", "Git", "Testing"],
        "url": "https://www.linkedin.com/jobs/"
    }
]

# Root & Auth Endpoints
@app.get("/", response_class=HTMLResponse)
def home():
    index_file = APP_DIR / "static" / "index.html"
    if not index_file.is_file():
        index_file = APP_DIR / "index.html"
    if not index_file.is_file():
        raise HTTPException(500, "Frontend index.html is missing.")
    return index_file.read_text(encoding="utf-8")

@app.post("/api/register")
def register(data: AuthIn):
    if len(data.password) < 8:
        raise HTTPException(400, "Password must be at least 8 characters.")
    email = data.email.strip().lower()
    if not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email):
        raise HTTPException(400, "Enter a valid email.")
    try:
        with db() as c:
            cur = c.execute(
                "INSERT INTO users(name,email,password_hash,created_at) VALUES(?,?,?,?)",
                (data.name.strip() or email.split("@")[0], email, pwd_context.hash(data.password), datetime.utcnow().isoformat())
            )
            uid = cur.lastrowid
            share_token = skills_passport.generate_share_token()
            c.execute(
                "INSERT INTO profiles(user_id,public_share_token,updated_at) VALUES(?,?,?)",
                (uid, share_token, datetime.utcnow().isoformat())
            )
            token = create_session(uid, c)
        return {
            "user_id": uid,
            "session_token": token,
            "name": data.name.strip() or email.split("@")[0],
            "email": email
        }
    except sqlite3.IntegrityError:
        raise HTTPException(409, "An account with this email already exists. Sign in instead.")

@app.post("/api/login")
def login(data: AuthIn):
    with db() as c:
        row = c.execute("SELECT * FROM users WHERE email=?", (data.email.strip().lower(),)).fetchone()
    if not row or not pwd_context.verify(data.password, row["password_hash"]):
        raise HTTPException(401, "Incorrect email or password.")
    token = create_session(row["id"])
    return {
        "user_id": row["id"],
        "session_token": token,
        "name": row["name"],
        "email": row["email"]
    }

@app.post("/api/logout")
def logout(request: Request):
    auth_header = request.headers.get("Authorization")
    token = auth_header[7:].strip() if auth_header and auth_header.startswith("Bearer ") else request.headers.get("X-Session-Token")
    if token:
        with db() as c:
            c.execute("DELETE FROM sessions WHERE token=?", (token,))
    return {"ok": True, "message": "Logged out successfully."}

@app.get("/api/me")
def me(request: Request):
    uid = user_from_request(request)
    with db() as c:
        u = c.execute("SELECT id, name, email, created_at FROM users WHERE id=?", (uid,)).fetchone()
        p = c.execute("SELECT * FROM profiles WHERE user_id=?", (uid,)).fetchone()
        if not p["public_share_token"]:
            st = skills_passport.generate_share_token()
            c.execute("UPDATE profiles SET public_share_token=? WHERE user_id=?", (st, uid))
            p = c.execute("SELECT * FROM profiles WHERE user_id=?", (uid,)).fetchone()
    return {"user": dict(u), "profile": dict(p)}

@app.put("/api/profile")
def save_profile(data: ProfileIn, request: Request):
    uid = user_from_request(request)
    role = data.target_role if data.target_role in ROLE_SKILLS else "Python Developer"
    with db() as c:
        c.execute("UPDATE users SET name=? WHERE id=?", (data.name.strip() or "Career Explorer", uid))
        c.execute(
            """UPDATE profiles SET education=?,experience=?,skills=?,target_role=?,location=?,work_mode=?,updated_at=? WHERE user_id=?""",
            (data.education, data.experience, data.skills, role, data.location, data.work_mode, datetime.utcnow().isoformat(), uid)
        )
    return {"ok": True, "message": "Career Digital Twin updated."}

@app.post("/api/resume")
async def resume_upload(request: Request, file: UploadFile = File(...)):
    uid = user_from_request(request)
    name = file.filename or "resume"
    raw = await file.read()
    if len(raw) > 8 * 1024 * 1024:
        raise HTTPException(413, "Maximum resume size is 8 MB.")
    text = ""
    try:
        if name.lower().endswith(".pdf"):
            import fitz
            doc = fitz.open(stream=raw, filetype="pdf")
            text = "\n".join(page.get_text() for page in doc)
        elif name.lower().endswith(".docx"):
            from docx import Document
            import io
            doc = Document(io.BytesIO(raw))
            text = "\n".join(p.text for p in doc.paragraphs)
        elif name.lower().endswith((".txt", ".md")):
            text = raw.decode("utf-8", errors="ignore")
        else:
            raise HTTPException(400, "Upload a PDF, DOCX, TXT or MD resume.")
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(400, "Could not read this file. Try a text-based PDF or DOCX.")

    # Rule-based extract
    skills = extract_skills(text)
    
    # NLP Semantic analysis of resume
    embedder = nlp_engine.get_embedder()
    res_vec = embedder.encode(text)
    
    with db() as c:
        p = c.execute("SELECT skills FROM profiles WHERE user_id=?", (uid,)).fetchone()
        existing = extract_skills((p["skills"] or "") + "\n" + text)
        c.execute(
            "UPDATE profiles SET resume_text=?,skills=?,updated_at=? WHERE user_id=?",
            (text, ", ".join(existing), datetime.utcnow().isoformat(), uid)
        )
        
    return {
        "filename": name,
        "characters": len(text),
        "skills": existing,
        "preview": text[:1200],
        "vector_dimensions": len(res_vec),
        "message": "Resume uploaded, NLP vector embeddings generated, and Career Digital Twin updated."
    }

@app.get("/api/roles")
def roles():
    return {"roles": [{"name": r, "description": ROLE_DESCRIPTIONS[r], "skills": ROLE_SKILLS[r]} for r in ROLE_SKILLS]}

@app.get("/api/analysis")
def analysis(request: Request):
    uid = user_from_request(request)
    with db() as c:
        p = dict(c.execute("SELECT * FROM profiles WHERE user_id=?", (uid,)).fetchone())
        projects = [dict(x) for x in c.execute("SELECT * FROM projects WHERE user_id=? ORDER BY id DESC", (uid,)).fetchall()]
        custom_tasks = [dict(x) for x in c.execute("SELECT * FROM roadmap_tasks WHERE user_id=? ORDER BY id DESC", (uid,)).fetchall()]
    
    full_text = (p.get("skills") or "") + "\n" + (p.get("resume_text") or "")
    skills = set(extract_skills(full_text))
    for pr in projects:
        skills.update(extract_skills(pr["skills"] + " " + pr["description"]))
        
    target = p.get("target_role") or "Python Developer"
    required = ROLE_SKILLS.get(target, ROLE_SKILLS["Python Developer"])
    matched = [s for s in required if s.lower() in {x.lower() for x in skills}]
    missing = [s for s in required if s not in matched]
    pct = round(100 * len(matched) / max(1, len(required)))
    
    # NLP semantic similarity between candidate resume and target role description
    role_desc = ROLE_DESCRIPTIONS.get(target, "")
    nlp_match = nlp_engine.match_resume_to_job(
        resume_text=p.get("resume_text") or p.get("experience") or "General experience",
        job_description=role_desc,
        required_skills=required,
        resume_skills=list(skills),
        weight_semantic=0.5,
        weight_keyword=0.5
    )
    
    roadmap = []
    for i, s in enumerate(missing[:6]):
        roadmap.append({
            "week": i + 1,
            "skill": s,
            "task": f"Study {s} fundamentals, complete one guided exercise, and document architecture decisions.",
            "proof": f"Create a small portfolio artifact demonstrating {s} in code."
        })
        
    return {
        "target_role": target,
        "description": role_desc,
        "skills": sorted(skills),
        "required": required,
        "matched": matched,
        "missing": missing,
        "match_percent": pct,
        "semantic_similarity_percent": nlp_match["semantic_score_pct"],
        "hybrid_fit_score": nlp_match["hybrid_score"],
        "roadmap": roadmap,
        "custom_tasks": custom_tasks,
        "semantic_bridges": nlp_match["semantic_bridges"],
        "explanation": f"{len(matched)} of {len(required)} target-role skills were found. NLP semantic embedding analysis indicates {nlp_match['semantic_score_pct']}% conceptual alignment."
    }

# Review 3 Core: Opportunities & Semantic Resume-Job Matching
@app.get("/api/jobs")
def get_jobs(request: Request, weight_semantic: float = 0.5, weight_keyword: float = 0.5):
    uid = user_from_request(request)
    with db() as c:
        p = dict(c.execute("SELECT * FROM profiles WHERE user_id=?", (uid,)).fetchone())
        projects = [dict(x) for x in c.execute("SELECT * FROM projects WHERE user_id=?", (uid,)).fetchall()]
        
    resume_text = p.get("resume_text") or (p.get("experience", "") + " " + p.get("education", ""))
    skills = set(extract_skills((p.get("skills") or "") + "\n" + resume_text))
    for pr in projects:
        skills.update(extract_skills(pr["skills"] + " " + pr["description"]))
        
    out = []
    for item in JOB_FEED:
        match_info = nlp_engine.match_resume_to_job(
            resume_text=resume_text,
            job_description=item["description"],
            required_skills=item["required_skills"],
            resume_skills=list(skills),
            weight_semantic=weight_semantic,
            weight_keyword=weight_keyword
        )
        out.append({
            "id": item["id"],
            "company": item["company"],
            "role": item["role"],
            "category": item["category"],
            "location": item["location"],
            "work_mode": item["work_mode"],
            "salary_range": item["salary_range"],
            "description": item["description"],
            "required_skills": item["required_skills"],
            "url": item["url"],
            "hybrid_score": match_info["hybrid_score"],
            "semantic_score_pct": match_info["semantic_score_pct"],
            "skill_overlap_pct": match_info["skill_overlap_pct"],
            "matched_skills": match_info["matched_skills"],
            "missing_skills": match_info["missing_skills"],
            "semantic_bridges": match_info["semantic_bridges"],
            "matched_concepts": match_info["matched_concepts"],
            "explanation": match_info["explanation"]
        })
        
    out.sort(key=lambda x: x["hybrid_score"], reverse=True)
    return {
        "jobs": out,
        "weights_applied": {"semantic": weight_semantic, "keyword": weight_keyword},
        "total_jobs": len(out),
        "note": "Opportunities scored using Review 3 Semantic NLP Embeddings & Cosine Similarity combined with explicit skill overlap."
    }

@app.post("/api/jobs/custom-match")
def custom_job_match(data: CustomMatchIn, request: Request):
    """
    Review 3 Proposed Implementation:
    Allows user to paste ANY job description and get an instant real-time
    vector similarity calculation, explicit skill overlap, and semantic explanation.
    """
    uid = user_from_request(request)
    with db() as c:
        p = dict(c.execute("SELECT * FROM profiles WHERE user_id=?", (uid,)).fetchone())
        projects = [dict(x) for x in c.execute("SELECT * FROM projects WHERE user_id=?", (uid,)).fetchall()]
        
    resume_text = p.get("resume_text") or (p.get("experience", "") + " " + p.get("education", ""))
    skills = set(extract_skills((p.get("skills") or "") + "\n" + resume_text))
    for pr in projects:
        skills.update(extract_skills(pr["skills"] + " " + pr["description"]))
        
    # Extract required skills from description if not explicitly provided
    if data.required_skills:
        req_skills = [s.strip() for s in data.required_skills.split(",") if s.strip()]
    else:
        req_skills = extract_skills(data.job_description)
        if not req_skills:
            req_skills = ["Python", "SQL", "Git"]
            
    match_result = nlp_engine.match_resume_to_job(
        resume_text=resume_text,
        job_description=data.job_description,
        required_skills=req_skills,
        resume_skills=list(skills),
        weight_semantic=data.weight_semantic or 0.5,
        weight_keyword=data.weight_keyword or 0.5
    )
    
    return {
        "job_title": data.job_title,
        "company": data.company,
        "required_skills": req_skills,
        "match_result": match_result
    }

@app.get("/api/evaluation/benchmarks")
def evaluation_benchmarks():
    """Returns Review 3 Benchmark evaluation suite comparing keyword vs semantic similarity."""
    return {"benchmarks": nlp_engine.get_evaluation_benchmarks()}

# Feature 8: Personal Career AI Agent
@app.post("/api/agent/chat")
def agent_chat(data: AgentChatIn, request: Request):
    uid = user_from_request(request)
    with db() as c:
        p = dict(c.execute("SELECT * FROM profiles WHERE user_id=?", (uid,)).fetchone())
        u = dict(c.execute("SELECT name FROM users WHERE id=?", (uid,)).fetchone())
        projects = [dict(x) for x in c.execute("SELECT * FROM projects WHERE user_id=?", (uid,)).fetchall()]
        interviews = [dict(x) for x in c.execute("SELECT * FROM interview_results WHERE user_id=?", (uid,)).fetchall()]
        apps = [dict(x) for x in c.execute("SELECT * FROM applications WHERE user_id=?", (uid,)).fetchall()]
        
    user_skills = extract_skills((p.get("skills") or "") + "\n" + (p.get("resume_text") or ""))
    target_role = p.get("target_role") or "Python Developer"
    req_skills = ROLE_SKILLS.get(target_role, ROLE_SKILLS["Python Developer"])
    matched = [s for s in req_skills if s in user_skills]
    missing = [s for s in req_skills if s not in user_skills]
    
    passport_data = skills_passport.calculate_skills_passport(p, projects, interviews)
    graph_data = skill_graph_data.analyze_skill_dependency_graph(user_skills, target_role)
    
    agent_output = career_agent.run_agent_reasoning(
        prompt=data.message,
        user_name=u.get("name", "Explorer"),
        target_role=target_role,
        skills=user_skills,
        matched_skills=matched,
        missing_skills=missing,
        projects=projects,
        interviews=interviews,
        applications=apps,
        roadmap=[],
        passport_data=passport_data,
        bottleneck_skill=graph_data.get("highest_leverage_skill") or {}
    )
    
    # Save conversation
    now = datetime.utcnow().isoformat()
    with db() as c:
        c.execute("INSERT INTO agent_messages(user_id, sender, message, actionable_tasks, created_at) VALUES(?,?,?,?,?)",
                  (uid, "user", data.message, "[]", now))
        c.execute("INSERT INTO agent_messages(user_id, sender, message, actionable_tasks, created_at) VALUES(?,?,?,?,?)",
                  (uid, "agent", agent_output["reply"], json.dumps(agent_output["actionable_tasks"]), now))
                  
    return agent_output

@app.get("/api/agent/history")
def agent_history(request: Request):
    uid = user_from_request(request)
    with db() as c:
        rows = c.execute("SELECT sender, message, actionable_tasks, created_at FROM agent_messages WHERE user_id=? ORDER BY id ASC LIMIT 50", (uid,)).fetchall()
    history = []
    for r in rows:
        tasks = []
        try:
            tasks = json.loads(r["actionable_tasks"])
        except Exception:
            pass
        history.append({
            "sender": r["sender"],
            "message": r["message"],
            "actionable_tasks": tasks,
            "created_at": r["created_at"]
        })
    return {"history": history}

@app.post("/api/roadmap/add-task")
def add_roadmap_task(data: TaskIn, request: Request):
    uid = user_from_request(request)
    with db() as c:
        cur = c.execute(
            "INSERT INTO roadmap_tasks(user_id, title, skill, category, status, created_at) VALUES(?,?,?,?,?,?)",
            (uid, data.title, data.skill or "", data.category or "Learning", "Pending", datetime.utcnow().isoformat())
        )
    return {"id": cur.lastrowid, "ok": True, "message": "Task added to your learning roadmap."}

# Feature 9: Skill Dependency Graph & Bottleneck Analysis
@app.get("/api/skill-graph")
def get_skill_graph(request: Request):
    uid = user_from_request(request)
    with db() as c:
        p = dict(c.execute("SELECT * FROM profiles WHERE user_id=?", (uid,)).fetchone())
    user_skills = extract_skills((p.get("skills") or "") + "\n" + (p.get("resume_text") or ""))
    target_role = p.get("target_role") or "Python Developer"
    analysis = skill_graph_data.analyze_skill_dependency_graph(user_skills, target_role)
    return analysis

# Feature 10: Evidence-Based Skills Passport
@app.get("/api/passport")
def get_passport(request: Request):
    uid = user_from_request(request)
    with db() as c:
        p = dict(c.execute("SELECT * FROM profiles WHERE user_id=?", (uid,)).fetchone())
        u = dict(c.execute("SELECT name, email FROM users WHERE id=?", (uid,)).fetchone())
        projects = [dict(x) for x in c.execute("SELECT * FROM projects WHERE user_id=?", (uid,)).fetchall()]
        interviews = [dict(x) for x in c.execute("SELECT * FROM interview_results WHERE user_id=?", (uid,)).fetchall()]
        
    passport = skills_passport.calculate_skills_passport(p, projects, interviews)
    share_token = p.get("public_share_token")
    if not share_token:
        share_token = skills_passport.generate_share_token()
        with db() as c:
            c.execute("UPDATE profiles SET public_share_token=? WHERE user_id=?", (share_token, uid))
            
    base_url = str(request.base_url).rstrip("/")
    public_url = f"{base_url}/passport/{share_token}"
    
    return {
        "candidate_name": u.get("name"),
        "target_role": p.get("target_role", "Python Developer"),
        "location": p.get("location", "India"),
        "public_url": public_url,
        "is_public": bool(p.get("is_public", 1)),
        "share_token": share_token,
        "passport": passport,
        "projects_count": len(projects),
        "interviews_count": len(interviews)
    }

@app.post("/api/passport/toggle-share")
def toggle_passport_share(request: Request):
    uid = user_from_request(request)
    with db() as c:
        p = dict(c.execute("SELECT is_public, public_share_token FROM profiles WHERE user_id=?", (uid,)).fetchone())
        new_val = 0 if p.get("is_public", 1) else 1
        c.execute("UPDATE profiles SET is_public=? WHERE user_id=?", (new_val, uid))
    return {"is_public": bool(new_val), "message": "Public sharing updated."}

# Public Recruiter View (No Login Required)
@app.get("/api/public/passport/{share_token}")
def public_passport_data(share_token: str):
    with db() as c:
        p = c.execute("SELECT * FROM profiles WHERE public_share_token=?", (share_token,)).fetchone()
        if not p or not p["is_public"]:
            raise HTTPException(404, "This candidate skills passport is either private or does not exist.")
        uid = p["user_id"]
        u = c.execute("SELECT name, email, created_at FROM users WHERE id=?", (uid,)).fetchone()
        projects = [dict(x) for x in c.execute("SELECT id, title, description, skills, live_url, github_url, created_at FROM projects WHERE user_id=?", (uid,)).fetchall()]
        interviews = [dict(x) for x in c.execute("SELECT role, question, score, created_at FROM interview_results WHERE user_id=? ORDER BY id DESC LIMIT 5", (uid,)).fetchall()]
        
    passport = skills_passport.calculate_skills_passport(dict(p), projects, interviews)
    return {
        "candidate": {
            "name": u["name"],
            "target_role": p["target_role"],
            "location": p["location"],
            "work_mode": p["work_mode"],
            "education": p["education"],
            "experience": p["experience"]
        },
        "passport": passport,
        "projects": projects,
        "interview_highlights": interviews
    }

@app.get("/passport/{share_token}", response_class=HTMLResponse)
def public_passport_page(share_token: str):
    """Clean, professional public recruiter view for verified candidate skills."""
    with db() as c:
        p = c.execute("SELECT * FROM profiles WHERE public_share_token=?", (share_token,)).fetchone()
        if not p or not p["is_public"]:
            return HTMLResponse("<h2>Portfolio not found or set to private by candidate.</h2>", status_code=404)
        uid = p["user_id"]
        u = c.execute("SELECT name FROM users WHERE id=?", (uid,)).fetchone()
        projects = [dict(x) for x in c.execute("SELECT * FROM projects WHERE user_id=?", (uid,)).fetchall()]
        interviews = [dict(x) for x in c.execute("SELECT * FROM interview_results WHERE user_id=?", (uid,)).fetchall()]
        
    passport = skills_passport.calculate_skills_passport(dict(p), projects, interviews)
    
    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{u['name']} — Verified Skills Passport | CareerIQ X</title>
<style>
:root{{--navy:#101c35;--blue:#3859d6;--ink:#19243a;--muted:#6f7b91;--line:#e6eaf1;--bg:#f5f7fb;--white:#fff;--green:#16856b;--amber:#b86a12}}
body{{margin:0;background:var(--bg);font:15px/1.5 Inter,ui-sans-serif,system-ui,sans-serif;color:var(--ink)}}
.wrap{{max-width:980px;margin:30px auto;padding:0 20px}}
.header{{background:linear-gradient(145deg,#0d1730,#1b3260);color:white;padding:36px;border-radius:20px;display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:20px}}
.badge{{display:inline-flex;padding:5px 12px;border-radius:99px;font-size:12px;font-weight:700;margin:3px}}
.badge.green{{background:#e3f6ef;color:#11765d}}
.badge.blue{{background:#e8edff;color:#354fb7}}
.badge.amber{{background:#fff2dc;color:#9c5c0c}}
.badge.muted{{background:#edf0f5;color:#6d7b92}}
.card{{background:white;border:1px solid var(--line);border-radius:16px;padding:24px;margin-top:20px;box-shadow:0 4px 18px #14234a04}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:16px}}
.score{{font-size:42px;font-weight:850;color:#3859d6}}
.skill-item{{padding:14px;border:1px solid #edf0f5;border-radius:12px;margin-bottom:10px;display:flex;justify-content:space-between;align-items:flex-start}}
</style></head>
<body>
<div class="wrap">
  <div class="header">
    <div>
      <div style="font-size:12px;text-transform:uppercase;letter-spacing:1px;color:#9cb5f3">Verified Career Passport</div>
      <h1 style="margin:6px 0">{u['name']}</h1>
      <div style="color:#d5e1fc">{p['target_role']} · {p['location'] or 'India'}</div>
    </div>
    <div style="text-align:right">
      <div style="font-size:12px;color:#9cb5f3">Evidence Trust Index</div>
      <div style="font-size:38px;font-weight:900;color:white">{passport['trust_index']}%</div>
      <div style="font-size:12px;color:#9ef0db">● {passport['credibility_status']}</div>
    </div>
  </div>
  
  <div class="card">
    <h3 style="margin-top:0">Verified Skills Breakdown ({passport['total_skills']} Skills)</h3>
    <p style="color:var(--muted);font-size:13px">Each skill status is backed by verifiable code artifacts, technical interview evaluations, or parsed resume evidence.</p>
    <div class="grid">
      { "".join(f'''<div class="skill-item">
          <div>
            <strong>{s['skill']}</strong>
            <div style="font-size:12px;color:var(--muted);margin-top:4px">{s['evidence_detail']}</div>
          </div>
          <span class="badge {s['badge_color']}">{s['tier']}</span>
        </div>''' for s in passport['passport_skills']) }
    </div>
  </div>

  <div class="card">
    <h3 style="margin-top:0">Evidence Portfolio & Projects ({len(projects)})</h3>
    { "".join(f'''<div style="border-bottom:1px solid #edf0f5;padding:14px 0">
        <h4 style="margin:0 0 6px">{pr['title']}</h4>
        <p style="margin:0 0 8px;color:#4a5670;font-size:13px">{pr['description']}</p>
        <div style="font-size:12px;color:var(--muted)">Demonstrated skills: {pr['skills']}</div>
      </div>''' for pr in projects) if projects else '<p style="color:var(--muted)">No public projects attached yet.</p>' }
  </div>
  <p style="text-align:center;font-size:12px;color:var(--muted);margin-top:28px">Verified by CareerIQ X Career Operating System</p>
</div>
</body></html>"""
    return html

# Privacy Controls
@app.get("/api/privacy/export")
def privacy_export(request: Request):
    uid = user_from_request(request)
    with db() as c:
        u = dict(c.execute("SELECT id, name, email, created_at FROM users WHERE id=?", (uid,)).fetchone())
        p = dict(c.execute("SELECT * FROM profiles WHERE user_id=?", (uid,)).fetchone())
        projects = [dict(x) for x in c.execute("SELECT * FROM projects WHERE user_id=?", (uid,)).fetchall()]
        apps = [dict(x) for x in c.execute("SELECT * FROM applications WHERE user_id=?", (uid,)).fetchall()]
        interviews = [dict(x) for x in c.execute("SELECT * FROM interview_results WHERE user_id=?", (uid,)).fetchall()]
        tasks = [dict(x) for x in c.execute("SELECT * FROM roadmap_tasks WHERE user_id=?", (uid,)).fetchall()]
    return {
        "account": u,
        "profile": p,
        "projects": projects,
        "applications": apps,
        "interview_practice": interviews,
        "roadmap_tasks": tasks,
        "export_timestamp": datetime.utcnow().isoformat(),
        "notice": "This export contains all personal career data stored locally for your account."
    }

@app.post("/api/privacy/anonymize")
def privacy_anonymize(request: Request):
    uid = user_from_request(request)
    with db() as c:
        p = c.execute("SELECT resume_text FROM profiles WHERE user_id=?", (uid,)).fetchone()
        raw = p["resume_text"] or ""
        # Redact emails
        redacted = re.sub(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", "[REDACTED_EMAIL]", raw)
        # Redact phone numbers
        redacted = re.sub(r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b", "[REDACTED_PHONE]", redacted)
        c.execute("UPDATE profiles SET resume_text=? WHERE user_id=?", (redacted, uid))
    return {"ok": True, "message": "Personal identifiable information (emails, phone numbers) redacted from stored resume."}

@app.delete("/api/privacy/account")
def privacy_delete_account(request: Request):
    uid = user_from_request(request)
    with db() as c:
        c.execute("DELETE FROM users WHERE id=?", (uid,))
    return {"ok": True, "message": "Account and all associated career data permanently deleted."}

# Projects & Evidence
@app.get("/api/projects")
def list_projects(request: Request):
    uid = user_from_request(request)
    with db() as c:
        rows = c.execute("SELECT * FROM projects WHERE user_id=? ORDER BY id DESC", (uid,)).fetchall()
    return {"projects": [dict(r) for r in rows]}

@app.post("/api/projects")
def add_project(data: ProjectIn, request: Request):
    uid = user_from_request(request)
    if not data.title.strip():
        raise HTTPException(400, "Project title is required.")
    with db() as c:
        cur = c.execute(
            "INSERT INTO projects(user_id,title,description,skills,live_url,github_url,created_at) VALUES(?,?,?,?,?,?,?)",
            (uid, data.title.strip(), data.description, data.skills, data.live_url or "", data.github_url or "", datetime.utcnow().isoformat())
        )
    return {"id": cur.lastrowid, "ok": True}

@app.delete("/api/projects/{pid}")
def delete_project(pid: int, request: Request):
    uid = user_from_request(request)
    with db() as c:
        c.execute("DELETE FROM projects WHERE id=? AND user_id=?", (pid, uid))
    return {"ok": True}

# Applications Tracker
@app.get("/api/applications")
def list_apps(request: Request):
    uid = user_from_request(request)
    with db() as c:
        rows = c.execute("SELECT * FROM applications WHERE user_id=? ORDER BY id DESC", (uid,)).fetchall()
    return {"applications": [dict(r) for r in rows]}

@app.post("/api/applications")
def add_app(data: ApplicationIn, request: Request):
    uid = user_from_request(request)
    if not data.company.strip() or not data.role.strip():
        raise HTTPException(400, "Company and role are required.")
    now = datetime.utcnow().isoformat()
    with db() as c:
        cur = c.execute(
            "INSERT INTO applications(user_id,company,role,status,url,notes,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?)",
            (uid, data.company, data.role, data.status, data.url, data.notes, now, now)
        )
    return {"id": cur.lastrowid, "ok": True}

@app.put("/api/applications/{aid}")
def update_app(aid: int, data: ApplicationUpdate, request: Request):
    uid = user_from_request(request)
    with db() as c:
        cur = c.execute(
            "UPDATE applications SET company=?,role=?,status=?,url=?,notes=?,updated_at=? WHERE id=? AND user_id=?",
            (data.company, data.role, data.status, data.url, data.notes, datetime.utcnow().isoformat(), aid, uid)
        )
        if cur.rowcount == 0:
            raise HTTPException(404, "Application not found.")
    return {"ok": True}

@app.delete("/api/applications/{aid}")
def delete_app(aid: int, request: Request):
    uid = user_from_request(request)
    with db() as c:
        c.execute("DELETE FROM applications WHERE id=? AND user_id=?", (aid, uid))
    return {"ok": True}

# Multi-Dimensional Interview Studio
@app.get("/api/interview/questions")
def interview_questions(request: Request, role: Optional[str] = None):
    uid = user_from_request(request)
    with db() as c:
        p = dict(c.execute("SELECT * FROM profiles WHERE user_id=?", (uid,)).fetchone())
    role = role or p.get("target_role") or "Python Developer"
    skills = ROLE_SKILLS.get(role, ROLE_SKILLS["Python Developer"])
    questions = [
        f"Introduce yourself and explain why you are targeting {role}.",
        f"Describe a real-world project where you implemented {skills[0]}. What specific architectural decisions did you make?",
        f"How would you demonstrate your proficiency in {skills[min(1, len(skills) - 1)]} when building a production service?",
        "Tell me about a difficult technical bug or system bottleneck you solved. What was your diagnosis methodology?",
        "How do you test and validate your software to ensure reliability, security, and performance?",
        "Describe a situation where you had to quickly learn an unfamiliar technology or framework to deliver a feature."
    ]
    return {"role": role, "questions": questions}

@app.post("/api/interview/answer")
def interview_answer(data: InterviewIn, request: Request):
    uid = user_from_request(request)
    answer = data.answer.strip()
    words = re.findall(r"\b[\w'-]+\b", answer)
    low = answer.lower()
    
    # 4-Dimensional STAR Rubric
    # 1. Situation & Context (25 pts)
    situation_score = 0
    if len(words) >= 40:
        situation_score += 15
    elif len(words) >= 20:
        situation_score += 10
    if any(k in low for k in ["problem", "project", "challenge", "company", "team", "client", "system", "requirement"]):
        situation_score += 10
        
    # 2. Personal Action & Contribution (25 pts)
    action_score = 0
    action_verbs = ["i built", "i designed", "i developed", "i implemented", "i analyzed", "i tested", "i configured", "i refactored", "i architected"]
    action_count = sum(1 for v in action_verbs if v in low)
    if action_count >= 2:
        action_score = 25
    elif action_count == 1:
        action_score = 18
    elif "i " in low:
        action_score = 10
        
    # 3. Measurable Result & Outcome (25 pts)
    result_score = 0
    if any(k in low for k in ["result", "outcome", "improved", "reduced", "increased", "achieved", "learned", "optimized", "%", "latency", "seconds"]):
        result_score += 15
    if any(char.isdigit() for char in answer):
        result_score += 10
        
    # 4. Technical Depth & Terminology (25 pts)
    tech_score = 0
    tech_terms = ["api", "database", "sql", "python", "fastapi", "react", "docker", "test", "security", "function", "query", "cache", "git", "cloud", "model", "pipeline"]
    term_count = sum(1 for t in tech_terms if t in low)
    if term_count >= 3:
        tech_score = 25
    elif term_count >= 1:
        tech_score = 15
        
    total_score = situation_score + action_score + result_score + tech_score
    
    notes = []
    if situation_score < 20:
        notes.append("Provide clearer initial context: describe the specific project scope and technical problem.")
    if action_score < 20:
        notes.append("Emphasize your individual contribution using direct first-person action verbs ('I designed...', 'I implemented...').")
    if result_score < 20:
        notes.append("Quantify the impact of your solution (e.g. latency reduction, test coverage percentage, error rate drops).")
    if tech_score < 20:
        notes.append("Deepen technical rigor: mention specific libraries, protocols, or architecture patterns used.")
        
    feedback = " ".join(notes) if notes else "Strong answer! You articulated clear situation context, decisive personal actions, measurable outcomes, and relevant technical depth."
    
    star_breakdown = {
        "situation": situation_score,
        "action": action_score,
        "result": result_score,
        "technical_depth": tech_score
    }
    
    with db() as c:
        c.execute(
            "INSERT INTO interview_results(user_id,role,question,answer,score,feedback,star_scores,created_at) VALUES(?,?,?,?,?,?,?,?)",
            (uid, data.role, data.question, answer, total_score, feedback, json.dumps(star_breakdown), datetime.utcnow().isoformat())
        )
        
    return {
        "score": total_score,
        "feedback": feedback,
        "star_breakdown": star_breakdown,
        "rubric": "Evaluated across Situation Context (25%), Personal Action Verbs (25%), Quantified Results (25%), and Technical Depth (25%)."
    }

@app.get("/api/interview/history")
def interview_history(request: Request):
    uid = user_from_request(request)
    with db() as c:
        rows = c.execute("SELECT * FROM interview_results WHERE user_id=? ORDER BY id DESC LIMIT 20", (uid,)).fetchall()
    return {"results": [dict(r) for r in rows]}

# Dashboard Overview
@app.get("/api/dashboard")
def dashboard(request: Request):
    uid = user_from_request(request)
    with db() as c:
        p = dict(c.execute("SELECT * FROM profiles WHERE user_id=?", (uid,)).fetchone())
        projects = c.execute("SELECT COUNT(*) n FROM projects WHERE user_id=?", (uid,)).fetchone()["n"]
        apps = c.execute("SELECT COUNT(*) n FROM applications WHERE user_id=?", (uid,)).fetchone()["n"]
        interviews = c.execute("SELECT COUNT(*) n FROM interview_results WHERE user_id=?", (uid,)).fetchone()["n"]
        projects_list = [dict(x) for x in c.execute("SELECT * FROM projects WHERE user_id=?", (uid,)).fetchall()]
        interview_list = [dict(x) for x in c.execute("SELECT * FROM interview_results WHERE user_id=?", (uid,)).fetchall()]
        
    skills = extract_skills((p.get("skills") or "") + "\n" + p.get("resume_text", ""))
    passport = skills_passport.calculate_skills_passport(p, projects_list, interview_list)
    graph = skill_graph_data.analyze_skill_dependency_graph(skills, p.get("target_role", "Python Developer"))
    
    return {
        "name": p.get("name", ""),
        "target_role": p.get("target_role", "Python Developer"),
        "skill_count": len(skills),
        "projects": projects,
        "applications": apps,
        "interviews": interviews,
        "passport_trust_index": passport["trust_index"],
        "highest_leverage_skill": graph.get("highest_leverage_skill"),
        "profile_complete": bool(p.get("education") or p.get("skills") or p.get("resume_text"))
    }

STATIC_DIR = APP_DIR / "static"
if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
