"""
Database Schemas

Define your MongoDB collection schemas here using Pydantic models.
These schemas are used for data validation in your application.

Each Pydantic model represents a collection in your database.
Model name is converted to lowercase for the collection name:
- User -> "user" collection
- Product -> "product" collection
- BlogPost -> "blogs" collection
"""

from pydantic import BaseModel, Field, EmailStr
from typing import Optional, List, Dict, Any

# Core user schema to support premium access
class User(BaseModel):
    """
    Users collection schema
    Collection name: "user" (lowercase of class name)
    """
    email: EmailStr = Field(..., description="Email address")
    name: Optional[str] = Field(None, description="Full name")
    is_premium: bool = Field(False, description="Premium subscription flag")


class ResumeAnalysis(BaseModel):
    """
    Resume analyses collection schema
    Collection name: "resumeanalysis"
    """
    email: Optional[EmailStr] = Field(None, description="User email if provided")
    resume_text: str = Field(..., description="Raw resume text pasted by user")
    job_description: Optional[str] = Field(None, description="Target job description text")
    premium: bool = Field(False, description="Whether premium features were used for this analysis")

    # Computed results
    ats_score: int = Field(..., ge=0, le=100, description="Overall ATS compatibility score")
    keyword_match_rate: float = Field(..., ge=0, le=1, description="Ratio of matched keywords")
    matched_keywords: List[str] = Field(default_factory=list)
    missing_keywords: List[str] = Field(default_factory=list)

    readability: Dict[str, Any] = Field(default_factory=dict, description="Readability metrics")
    sections: Dict[str, Any] = Field(default_factory=dict, description="Section-by-section grades and notes")
    recommendations: List[str] = Field(default_factory=list, description="Actionable recommendations")
    highlights: List[str] = Field(default_factory=list, description="Suggested achievement-focused bullet points")


# Example schema retained for reference (not used by the app directly)
class Product(BaseModel):
    title: str
    description: Optional[str] = None
    price: float
    category: str
    in_stock: bool = True
