import os
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
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
    advanced: Optional[Dict[str, Any]] = None


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


def generate_advanced_metrics(text: str, jd: str) -> Dict[str, Any]:
    import re
    lower = text.lower()
    # Basic contact info
    emails = re.findall(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text)
    phones = re.findall(r"(?:(?:\+?\d{1,3}[\s.-]?)?(?:\(\d{3}\)|\d{3})[\s.-]?\d{3}[\s.-]?\d{4})", text)
    linkedin = re.findall(r"https?://(www\.)?linkedin\.com/[A-Za-z0-9_\-/]+", text, flags=re.I)

    # Skills dictionary (expandable)
    skills_bank = [
        'python','java','javascript','typescript','react','node','fastapi','django','flask','aws','gcp','azure','docker','kubernetes','sql','nosql','mongodb','postgres','redis','graphql','rest','ci','cd','linux','git','pandas','numpy','scikit','tensorflow','pytorch','spark','hadoop','airflow','kafka','terraform','ansible'
    ]
    skills_detected = sorted({s for s in skills_bank if s in lower})

    # Action verbs analysis
    verbs = [
        'led','owned','delivered','built','designed','implemented','optimized','improved','launched','migrated','automated','architected','scaled','mentored','managed','shipped','deployed','integrated','analyzed','refactored'
    ]
    strong_verbs_used = sorted({v for v in verbs if re.search(rf"\b{v}\b", lower)})
    action_verb_count = sum(len(re.findall(rf"\b{v}\b", lower)) for v in verbs)

    # Tense check: look for past tense vs present in bullets
    past = len(re.findall(r"\b(ed)\b", lower))
    present = len(re.findall(r"\b(ing)\b", lower))
    tense = "balanced" if abs(past-present) <= 5 else ("past-heavy" if past>present else "present-heavy")

    # Dates coverage and gaps
    dates = re.findall(r"(20\d{2}|19\d{2})", text)
    years = sorted({int(y) for y in dates})
    coverage = {"years_mentioned": years, "min_year": min(years) if years else None, "max_year": max(years) if years else None}
    gaps = []
    if len(years) >= 2:
        for a, b in zip(years, years[1:]):
            if b - a > 2:
                gaps.append({"from": a, "to": b, "gap_years": b - a})

    # Education/degree detection
    degrees = re.findall(r"(?i)(b\.?sc\.?|m\.?sc\.?|bachelor|master|ph\.?d\.?|mba|b\.?tech|m\.?tech)", text)

    # Bullet quality
    bullets = [line.strip() for line in text.splitlines() if line.strip().startswith(('-', '•', '*'))]
    avg_bullet_len = round(sum(len(b) for b in bullets)/max(1, len(bullets)), 1) if bullets else 0

    # JD alignment top missing
    jd_keywords = extract_keywords(jd) if jd else []
    resume_keywords = extract_keywords(text)
    top_missing = sorted(list(set(jd_keywords) - set(resume_keywords)))[:15]

    # Pages estimate
    pages = text.count('\f') or None

    return {
        "contact": {
            "emails": emails[:2],
            "phones": phones[:2],
            "linkedin": linkedin[:2]
        },
        "skills_detected": skills_detected,
        "action_verbs_count": action_verb_count,
        "strong_verbs_used": strong_verbs_used,
        "tense": tense,
        "date_coverage": coverage,
        "gaps": gaps,
        "degrees": list({d[0] if isinstance(d, tuple) else d for d in degrees})[:5] if degrees else [],
        "avg_bullet_length": avg_bullet_len,
        "top_missing_from_jd": top_missing,
        "pages": pages,
    }


def analyze_text(resume: str, jd: str, email: Optional[str], premium: bool) -> AnalyzeResponse:
    if not resume.strip():
        raise HTTPException(status_code=400, detail="Resume text is required")

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
    if premium:
        if jd_keywords:
            recs.append("Premium: Mirror phrasing from the job description for top keywords, where authentic.")
        recs.append("Premium: Quantify impact with concrete numbers (%, $, time saved).")
        recs.append("Premium: Add a tailored summary using the target role title.")

    advanced = generate_advanced_metrics(resume, jd) if premium else {}

    result = AnalyzeResponse(
        ats_score=score,
        keyword_match_rate=round(keyword_match_rate, 2),
        matched_keywords=matched,
        missing_keywords=missing,
        readability=read,
        sections=sections,
        recommendations=recs,
        highlights=highlights,
        advanced=advanced or None,
    )

    # Persist analysis
    try:
        _doc = ResumeAnalysis(
            email=email,
            resume_text=resume,
            job_description=jd or None,
            premium=premium,
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
        pass

    return result


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze_resume(payload: AnalyzeRequest):
    jd = (payload.job_description or "").strip()
    return analyze_text(payload.resume_text, jd, payload.email, payload.premium)


def extract_pdf_text(file_bytes: bytes) -> str:
    try:
        from pdfminer.high_level import extract_text
        import io
        return extract_text(io.BytesIO(file_bytes))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read PDF: {str(e)[:120]}")


@app.post("/analyze/pdf", response_model=AnalyzeResponse)
async def analyze_pdf(
    file: UploadFile = File(...),
    job_description: Optional[str] = Form(None),
    email: Optional[str] = Form(None),
    premium: Optional[bool] = Form(False),
):
    if file.content_type not in ("application/pdf", "application/octet-stream"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    content = await file.read()
    text = extract_pdf_text(content)
    jd = (job_description or "").strip()
    # premium comes as string sometimes
    if isinstance(premium, str):
        premium = premium.lower() in ("1","true","yes","on")
    return analyze_text(text, jd, email, bool(premium))


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
