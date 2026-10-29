# CareerIQ X — Review 3 Enhanced Edition

CareerIQ X is an intelligent, explainable Career Operating System designed to help engineering candidates analyze skill gaps, verify practical competencies, and match with opportunities using modern NLP vector embeddings.

---

## 🚀 Key Review 3 Enhancements

### 1. Semantic Resume–Job Matching Using NLP Embeddings
- **Mathematical Vector Embeddings**: Resumes and job descriptions are transformed into high-dimensional numerical vectors using our domain-trained NLP embedding model.
- **Cosine Similarity**:
  $$\text{CosSim}(A, B) = \frac{A \cdot B}{\|A\| \times \|B\|}$$
  Where $A$ is the resume embedding vector and $B$ is the job description vector.
- **Weighted Hybrid Scoring**:
  $$\text{Composite Fit} = (w_{\text{semantic}} \times \text{SemanticSimilarity}) + (w_{\text{skill}} \times \text{SkillOverlap})$$
  Includes an interactive weight tuning slider in the Opportunity Discovery UI.
- **Semantic Bridge Explanations**: Detects equivalent engineering concepts even when vocabulary differs (e.g., *"developed predictive models from historical data"* $\approx$ *"machine learning model development"*).
- **Custom Job Description Analyzer**: Paste any external job posting from LinkedIn, Indeed, or company career sites for real-time vector comparison and gap breakdown.
- **Review 3 Benchmark Suite**: Built-in evaluation dashboard comparing rule-based matching vs semantic embeddings across edge cases, false friends (e.g., Java vs JavaScript), and domain distinctions.

### 2. Personal Career AI Agent (Item 8)
- Context-aware career co-pilot embedded directly in your workspace.
- Ingests your Career Digital Twin, uploaded resume text, portfolio projects, interview results, and roadmap.
- Answers questions like:
  - *"What should I work on this week?"* $\rightarrow$ generates a tailored weekly plan with 3 actionable tasks.
  - *"What is my highest-leverage missing skill?"*
  - *"What project should I build to demonstrate my missing skills?"*
  - *"Evaluate my interview readiness."*
- One-click **"Commit to Roadmap"** button turns recommendations directly into active tasks.

### 3. Skill Dependency Graph & Bottleneck Analysis (Item 9)
- Interactive directed graph connecting roles, foundation skills, intermediate skills, advanced specializations, and prerequisites.
- **Bottleneck Analysis Algorithm**: Automatically calculates graph centrality and out-degree to identify the **#1 Highest-Leverage Missing Skill** that unlocks the most downstream competencies and career paths.
- Click any node to inspect prerequisites, unlocked downstream capabilities, and target role impacts.

### 4. Evidence-Based Skills Passport (Item 10)
- Moves beyond unverified resume claims to a 4-tier evidence verification model:
  - 🟢 **Project-Evidenced (100% Trust)**: Backed by working code, GitHub repos, or live demo URLs.
  - 🔵 **Interview-Passed (75% Trust)**: Demonstrated in mock interview evaluations with passing rubric scores ($\ge 60\%$).
  - 🟡 **Resume-Detected (50% Trust)**: Extracted and identified from uploaded resume documents.
  - ⚪ **Self-Reported (25% Trust)**: Claimed in personal profile without external validation.
- Computes an overall **Passport Trust Index** percentage.
- **Public Recruiter Portfolio Link**: Generate a unique URL (`/passport/{share_token}`) that recruiters can view without needing an account.

### 5. Multi-Dimensional STAR Interview Studio
- Evaluates technical interview answers across four distinct dimensions (25 points each):
  1. **Situation & Context (25%)**: Scope, environment, and technical challenge.
  2. **Personal Action Verbs (25%)**: Explicit first-person contributions (*"I designed..."*, *"I implemented..."*).
  3. **Quantified Results (25%)**: Measured outcomes, latency reductions, accuracy percentages, or throughput metrics.
  4. **Technical Depth (25%)**: Protocols, libraries, database design, and architecture tradeoffs.

### 6. Strengthened Security & Privacy Controls
- **Cryptographic Session Management**: High-entropy 256-bit session tokens stored with expiration timestamps.
- **Password Security**: Passlib bcrypt password hashing.
- **Data Privacy**:
  - Full machine-readable account export as JSON (`/api/privacy/export`).
  - Automated Resume PII Sanitization/Redaction (`/api/privacy/anonymize`).
  - Permanent account deletion (`/api/privacy/account`).

---

## 💻 Running the Application

### macOS / Linux
```bash
chmod +x run_mac_linux.sh
./run_mac_linux.sh
```
Open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your web browser.

### Windows
Double-click `run_windows.bat` or run:
```bat
run_windows.bat
```
Then open [http://127.0.0.1:8000](http://127.0.0.1:8000).

### Running Tests
To run the automated verification test suite:
```bash
./.venv/bin/python test_review3_enhancement.py
```

### Re-training the Domain NLP Embedding Model
To re-train the semantic embedding model on custom domain corpora:
```bash
./.venv/bin/python train_domain_model.py
```
Model artifacts will be updated in `trained_models/domain_nlp_embedder.joblib`.
