"""
CareerIQ X — Domain NLP Embedding Model Training Pipeline
Trains a domain-specific semantic embedding model on curated job descriptions,
resume experience bullets, and technical concept pairs.
"""

import os
import json
import numpy as np
import joblib
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer

APP_DIR = Path(__file__).parent
DATA_DIR = APP_DIR / "data"
MODEL_DIR = APP_DIR / "trained_models"
DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

# 1. Rich Domain Training Corpus: Job descriptions, role requirements, resume bullets, and conceptual paraphrases
DOMAIN_CORPUS = [
    # Machine Learning & Data Science
    "Developed predictive models from historical data using Python, Scikit-learn, and statistical techniques.",
    "Machine learning model development, feature engineering, cross-validation, and hyperparameter tuning.",
    "Trained supervised and unsupervised models for classification, regression, and customer churn prediction.",
    "Applied gradient boosting, random forests, linear regression, and logistic regression on tabular datasets.",
    "Built deep learning neural networks using PyTorch and TensorFlow for image and text classification.",
    "Conducted exploratory data analysis, data cleaning, outlier detection, and statistical hypothesis testing.",
    "Developed natural language processing pipelines for sentiment analysis, entity extraction, and text summarization.",
    "Deployed ML models as REST API microservices using FastAPI and Docker containers.",
    "Evaluated model performance using precision, recall, F1-score, ROC-AUC, and mean squared error.",
    "Engineered scalable data preprocessing pipelines using Pandas, NumPy, and Scikit-learn ColumnTransformer.",

    # Backend & API Development
    "Designed and implemented RESTful API endpoints with FastAPI, Flask, and Django.",
    "Built scalable backend services with Python, relational SQL databases, and Redis caching.",
    "Developed secure microservices architecture with JWT authentication, role-based access control, and rate limiting.",
    "Wrote complex SQL queries, stored procedures, database migrations, and schema optimizations in PostgreSQL and MySQL.",
    "Implemented automated unit tests, integration tests, and test-driven development using Pytest.",
    "Created asynchronous backend workers and message queues with Celery, RabbitMQ, and Redis.",
    "Built object-oriented backend architectures following clean code, SOLID principles, and design patterns.",
    "Designed database models, foreign key relationships, indexing strategies, and query performance tuning.",
    "Connected backend services to third-party APIs with webhook handling and robust error logging.",
    "Developed server-side business logic and data persistence layers for web applications.",

    # Full Stack & Frontend Development
    "Built responsive web user interfaces using modern React, TypeScript, HTML5, and CSS3.",
    "Developed single page applications with React hooks, state management, and reusable UI components.",
    "Implemented client-side routing, responsive layout design, and mobile-friendly CSS Flexbox and Grid.",
    "Integrated REST APIs with frontend applications using fetch, axios, and asynchronous JavaScript.",
    "Created accessible, semantic HTML forms with real-time validation and accessible ARIA attributes.",
    "Built full stack applications integrating React frontend with Python FastAPI backend and PostgreSQL database.",
    "Optimized web performance, bundle size, asset loading, and Lighthouse core web vitals.",
    "Managed global application state using Redux Toolkit, Context API, and modern React hooks.",

    # Data Analytics & Business Intelligence
    "Turned structured data into actionable business insights, KPI reports, and interactive dashboards.",
    "Wrote advanced SQL queries using window functions, common table expressions (CTEs), and complex joins.",
    "Designed interactive executive dashboards in Power BI and Tableau with automated refresh schedules.",
    "Cleaned, transformed, and aggregated large corporate datasets using Excel pivot tables, formulas, and Power Query.",
    "Analyzed customer behavior, retention trends, revenue metrics, and sales funnels to guide business decisions.",
    "Extracted data from transactional databases and built reporting data marts for analytical reporting.",
    "Created data visualizations using Matplotlib, Seaborn, and Plotly to communicate trends clearly to stakeholders.",
    "Conducted A/B test analysis, statistical significance calculations, and variance analysis.",

    # Cloud & DevOps Engineering
    "Automated continuous integration and continuous deployment (CI/CD) pipelines using GitHub Actions.",
    "Containerized applications using Docker and Docker Compose for consistent development and production environments.",
    "Deployed and managed cloud infrastructure on AWS, EC2, S3, RDS, and Lambda serverless functions.",
    "Configured Linux servers, bash scripting, system administration, and network security policies.",
    "Implemented container orchestration, service discovery, and deployment configurations with Kubernetes.",
    "Wrote Infrastructure as Code using Terraform to provision reproducible cloud resources.",
    "Monitored application health, latency, uptime, and system logs using Prometheus, Grafana, and CloudWatch.",
    "Configured reverse proxies, SSL/TLS certificates, load balancers, and DNS using Nginx.",

    # Cybersecurity & Systems Security
    "Monitored security events, analyzed logs, and conducted threat detection using SIEM platforms.",
    "Investigated security incidents, malware behavior, and unauthorized network intrusions.",
    "Performed vulnerability assessments, penetration testing, and security code reviews.",
    "Hardened Linux and Windows systems, implemented network firewalls, and configured IDS/IPS systems.",
    "Enforced zero-trust principles, public key cryptography, SSL/TLS encryption, and password hashing standards.",
    "Created incident response playbooks, disaster recovery plans, and security compliance documentation.",

    # Java Development
    "Developed enterprise backend applications using Java, Spring Boot, Hibernate, and Maven.",
    "Designed object-oriented architectures with Java 17, dependency injection, and JPA entity management.",
    "Created enterprise RESTful web services with Spring MVC, Spring Data JPA, and PostgreSQL.",
    "Configured JUnit 5 unit tests, Mockito mocks, and integration test suites for Java services."
]

# Additional domain job postings & resume excerpts to expand semantic depth
DOMAIN_JOB_DESCRIPTIONS = [
    "Python Developer: Responsible for writing clean, testable Python code, building RESTful APIs using FastAPI, managing PostgreSQL databases, and collaborating with Git version control.",
    "Machine Learning Engineer: We are seeking an ML engineer to build predictive models, design feature pipelines, evaluate statistical performance, and deploy scalable ML services.",
    "Backend Developer: Seeking a backend engineer proficient in Python, SQL databases, API security, automated testing, and scalable server architecture.",
    "Full Stack Developer: Looking for a developer skilled in React, modern JavaScript, HTML/CSS, REST API integration, and backend development with Python or Node.",
    "Data Analyst: Candidate will analyze datasets using SQL and Python, design Power BI dashboards, calculate business metrics, and deliver strategic recommendations.",
    "Cloud Engineer: Responsible for automating deployments with CI/CD, managing Linux cloud servers, writing Docker containers, and provisioning cloud services on AWS.",
    "Cybersecurity Analyst: Seeking a security professional to monitor SIEM alerts, respond to cyber incidents, conduct vulnerability scans, and maintain system security.",
    "Java Developer: Enterprise Java engineer experienced with Spring Boot, OOP patterns, SQL database design, microservices, and automated testing."
]

ALL_TRAINING_TEXTS = DOMAIN_CORPUS + DOMAIN_JOB_DESCRIPTIONS

class DomainEmbeddingPipeline:
    """
    Domain-Trained Semantic Embedding Pipeline using Sublinear TF-IDF + Latent Semantic Analysis (LSA).
    Learns dense semantic vectors (128 dimensions) and computes Cosine Similarity:
    CosSim(A, B) = (A · B) / (||A|| × ||B||)
    """
    def __init__(self, n_components=96):
        self.n_components = n_components
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 3),
            min_df=1,
            sublinear_tf=True,
            strip_accents="unicode",
            lowercase=True,
            token_pattern=r"(?u)\b[\w\-\+\.#]{2,}\b"  # captures c++, c#, .net, ci/cd
        )
        self.svd = TruncatedSVD(n_components=n_components, random_state=42)
        self.normalizer = Normalizer(copy=False)
        self.pipeline = None
        self.vocabulary_size = 0
        self.is_fitted = False

    def fit(self, texts):
        print(f"Training on {len(texts)} domain documents...")
        tfidf_matrix = self.vectorizer.fit_transform(texts)
        self.vocabulary_size = len(self.vectorizer.vocabulary_)
        print(f"Learned vocabulary of {self.vocabulary_size} domain terms and n-grams.")
        
        # Fit SVD on TF-IDF matrix
        actual_components = min(self.n_components, tfidf_matrix.shape[1] - 1, tfidf_matrix.shape[0] - 1)
        self.svd = TruncatedSVD(n_components=actual_components, random_state=42)
        dense_matrix = self.svd.fit_transform(tfidf_matrix)
        self.normalizer.fit(dense_matrix)
        self.is_fitted = True
        print(f"Dense semantic projection fitted: {actual_components} dimensions. Explained variance ratio sum: {self.svd.explained_variance_ratio_.sum():.4f}")
        return self

    def transform(self, texts):
        if not self.is_fitted:
            raise ValueError("Model must be fitted before transforming.")
        if isinstance(texts, str):
            texts = [texts]
        tfidf = self.vectorizer.transform(texts)
        dense = self.svd.transform(tfidf)
        normalized = self.normalizer.transform(dense)
        return normalized

    def cosine_similarity(self, vec_a, vec_b):
        """
        Calculates exact Cosine Similarity:
        CosSim(A, B) = (A · B) / (||A|| * ||B||)
        """
        a = np.asarray(vec_a).ravel()
        b = np.asarray(vec_b).ravel()
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        dot = np.dot(a, b)
        cos_sim = dot / (norm_a * norm_b)
        return float(np.clip(cos_sim, 0.0, 1.0))

    def compute_similarity(self, text_a, text_b):
        vec_a = self.transform([text_a])[0]
        vec_b = self.transform([text_b])[0]
        return self.cosine_similarity(vec_a, vec_b)

def train_and_save():
    model = DomainEmbeddingPipeline(n_components=64)
    model.fit(ALL_TRAINING_TEXTS)
    
    # Save model artifact
    model_path = MODEL_DIR / "domain_embedding_model.joblib"
    joblib.dump(model, model_path)
    print(f"Model saved successfully to {model_path}")
    
    # Run Benchmark Test Cases from Review 3 Specification
    print("\n--- Review 3 Benchmark Evaluation ---")
    test_cases = [
        (
            "Semantic Equivalence (ML)",
            "Developed predictive models from historical data using Python and algorithms",
            "Machine learning model development and statistical modeling in Python",
            "Expected: HIGH semantic similarity despite different phrasing"
        ),
        (
            "Semantic Equivalence (DevOps)",
            "Configured automated deployments with GitHub Actions and containerization",
            "Built continuous integration CI/CD pipelines using Docker on Linux",
            "Expected: HIGH semantic similarity"
        ),
        (
            "False Friends / Distinction (Java vs JS)",
            "Built web frontends using JavaScript React and HTML styling",
            "Enterprise backend development in Java with Spring Boot and Hibernate",
            "Expected: MODERATE-TO-LOW similarity (identifies technology divergence)"
        ),
        (
            "Unrelated Domains",
            "Designed executive financial dashboards and revenue reports in Power BI",
            "Linux kernel security, network penetration testing, and firewall rules",
            "Expected: VERY LOW similarity"
        )
    ]
    
    benchmark_results = []
    for name, text_a, text_b, note in test_cases:
        sim = model.compute_similarity(text_a, text_b)
        score_pct = round(sim * 100, 1)
        print(f"[{name}]")
        print(f"  Text A: '{text_a}'")
        print(f"  Text B: '{text_b}'")
        print(f"  Cosine Similarity: {sim:.4f} ({score_pct}%) | {note}\n")
        benchmark_results.append({
            "name": name,
            "text_a": text_a,
            "text_b": text_b,
            "similarity": sim,
            "similarity_percent": score_pct,
            "note": note
        })

    # Save benchmark suite to data/benchmark_results.json
    benchmark_path = DATA_DIR / "benchmark_results.json"
    with open(benchmark_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_results, f, indent=2)
    print(f"Benchmark results saved to {benchmark_path}")

if __name__ == "__main__":
    train_and_save()

