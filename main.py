import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

from database import db, create_document, get_documents
from schemas import ResumeAnalysis, User

app = FastAPI(title="AI Resume Analyzer API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalyzeRequest(BaseModel):
    resume_text: str
    job_description: Optional[str] = None
    email: Optional[str] = None
    premium: bool = False


class AnalyzeResponse(BaseModel):
    ats_score: int
    keyword_match_rate: float
    matched_keywords: List[str]
    missing_keywords: List[str]
    readability: Dict[str, Any]
    sections: Dict[str, Any]
    recommendations: List[str]
    highlights: List[str]


@app.get("/")
def read_root():
    return {"message": "Resume Analyzer Backend Running"}


@app.get("/test")
def test_database():
    """Test endpoint to check if database is available and accessible"""
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Available",
        "database_url": None,
        "database_name": None,
        "connection_status": "Not Connected",
        "collections": []
    }

    try:
        if db is not None:
            response["database"] = "✅ Available"
            response["database_url"] = "✅ Configured"
            response["database_name"] = db.name if hasattr(db, 'name') else "✅ Connected"
            response["connection_status"] = "Connected"
            try:
                collections = db.list_collection_names()
                response["collections"] = collections[:10]
                response["database"] = "✅ Connected & Working"
            except Exception as e:
                response["database"] = f"⚠️  Connected but Error: {str(e)[:50]}"
        else:
            response["database"] = "⚠️  Available but not initialized"

    except Exception as e:
        response["database"] = f"❌ Error: {str(e)[:50]}"

    response["database_url"] = "✅ Set" if os.getenv("DATABASE_URL") else "❌ Not Set"
    response["database_name"] = "✅ Set" if os.getenv("DATABASE_NAME") else "❌ Not Set"

    return response


def extract_keywords(text: str) -> List[str]:
    import re
    words = re.findall(r"[A-Za-z][A-Za-z\-\+\/#]*", text.lower())
    stop = set([
        'the','and','a','an','to','of','in','on','for','with','is','are','as','by','at','from','that','this','it','be','or','your','you','our'
    ])
    keywords = [w for w in words if len(w) > 2 and w not in stop]
    # Deduplicate, keep order
    seen = set()
    uniq = []
    for w in keywords:
        if w not in seen:
            seen.add(w)
            uniq.append(w)
    return uniq[:200]


def simple_readability_metrics(text: str) -> Dict[str, Any]:
    sentences = max(1, text.count('.') + text.count('!') + text.count('?'))
    words = max(1, len(text.split()))
    chars = max(1, sum(len(w) for w in text.split()))
    avg_sentence_len = words / sentences
    avg_word_len = chars / words
    bullet_points = text.count('\n-') + text.count('\n•') + text.count('\n*')
    return {
        "sentences": sentences,
        "words": words,
        "avg_sentence_len": round(avg_sentence_len, 2),
        "avg_word_len": round(avg_word_len, 2),
        "bullet_points": bullet_points,
    }


def grade_sections(text: str) -> Dict[str, Any]:
    lower = text.lower()
    sections = {
        "summary": ("summary" in lower or "about" in lower),
        "experience": ("experience" in lower or "work history" in lower),
        "skills": ("skills" in lower or "technologies" in lower),
        "education": ("education" in lower),
        "projects": ("projects" in lower),
    }
    grades = {k: ("present" if v else "missing") for k, v in sections.items()}
    return grades


def generate_recommendations(missing_keywords: List[str], section_grades: Dict[str, Any]) -> List[str]:
    recs = []
    if missing_keywords:
        recs.append(f"Add these relevant keywords to improve ATS match: {', '.join(missing_keywords[:10])}")
    for section, status in section_grades.items():
        if status == "missing":
            recs.append(f"Add a clear {section.title()} section.")
    recs.append("Use concise bullet points starting with impact verbs and include metrics.")
    recs.append("Keep formatting simple: standard fonts, no tables or images for ATS.")
    recs.append("Tailor your summary with the job title and 2-3 signature strengths.")
    return recs


def suggest_highlights(resume_text: str) -> List[str]:
    # Heuristic placeholder suggestions derived from verbs + numbers
    import re
    highlights = []
    patterns = [
        r"(?i)(increased|reduced|improved|optimized|led|launched|built|designed|implemented) [^\n\.]{0,60} (by|to) \d+%",
        r"(?i)(saved|cut) costs [^\n\.]{0,60} \$?\d+[kKmM]?",
        r"(?i)managed [^\n\.]{0,60} team",
    ]
    for p in patterns:
        for m in re.findall(p, resume_text):
            if isinstance(m, tuple):
                highlights.append(" ".join(m))
            else:
                highlights.append(m)
    # Fallback generic suggestions
    if not highlights:
        highlights = [
            "Led cross-functional initiative delivering X% improvement in Y within Z months.",
            "Optimized process to reduce costs by $X or time by Y% across N teams.",
            "Built and deployed feature used by N users, improving retention by Y%.",
        ]
    return highlights[:5]


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze_resume(payload: AnalyzeRequest):
    resume = payload.resume_text.strip()
    if not resume:
        raise HTTPException(status_code=400, detail="Resume text is required")

    jd = (payload.job_description or "").strip()

    resume_keywords = extract_keywords(resume)
    jd_keywords = extract_keywords(jd) if jd else []

    matched = sorted(set(resume_keywords).intersection(set(jd_keywords))) if jd_keywords else []
    missing = sorted(set(jd_keywords) - set(resume_keywords)) if jd_keywords else []

    keyword_match_rate = (len(matched) / max(1, len(jd_keywords))) if jd_keywords else 0.0

    # Base ATS score factors
    score = 50
    score += int(keyword_match_rate * 40)  # up to +40 for keyword alignment
    read = simple_readability_metrics(resume)
    if read.get("bullet_points", 0) >= 3:
        score += 5
    if read.get("avg_sentence_len", 0) <= 25:
        score += 5
    score = max(0, min(100, score))

    sections = grade_sections(resume)
    recs = generate_recommendations(missing, sections)
    highlights = suggest_highlights(resume)

    # Premium extras: deeper checks and more suggestions
    if payload.premium:
        # Add a couple of extra heuristics for premium users
        if jd_keywords:
            recs.append("Premium: Mirror phrasing from the job description for top keywords, where authentic.")
        recs.append("Premium: Quantify impact with concrete numbers (%, $, time saved).")
        recs.append("Premium: Add a tailored summary using the target role title.")

    result = AnalyzeResponse(
        ats_score=score,
        keyword_match_rate=round(keyword_match_rate, 2),
        matched_keywords=matched,
        missing_keywords=missing,
        readability=read,
        sections=sections,
        recommendations=recs,
        highlights=highlights,
    )

    # Persist analysis
    try:
        _doc = ResumeAnalysis(
            email=payload.email,
            resume_text=resume,
            job_description=jd or None,
            premium=payload.premium,
            ats_score=result.ats_score,
            keyword_match_rate=result.keyword_match_rate,
            matched_keywords=result.matched_keywords,
            missing_keywords=result.missing_keywords,
            readability=result.readability,
            sections=result.sections,
            recommendations=result.recommendations,
            highlights=result.highlights,
        )
        create_document("resumeanalysis", _doc)
    except Exception:
        # Swallow DB errors to keep API responsive even if DB is down
        pass

    return result


@app.get("/history")
def get_history(email: Optional[str] = None, limit: int = 20):
    try:
        flt = {"email": email} if email else {}
        docs = get_documents("resumeanalysis", flt, limit)
        # Convert ObjectId and datetime for JSON friendliness
        def clean(d):
            d = dict(d)
            d.pop("_id", None)
            for k, v in list(d.items()):
                if hasattr(v, "isoformat"):
                    d[k] = v.isoformat()
            return d
        return [clean(x) for x in docs]
    except Exception:
        return []


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
