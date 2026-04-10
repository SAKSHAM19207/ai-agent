from openenv.core.env_server.types import Action, Observation
from pydantic import BaseModel, Field
from typing import List, Optional, Dict

class UserProfile(BaseModel):
    skills: List[str] = Field(default_factory=list)
    experience: int = 0
    preferred_roles: List[str] = Field(default_factory=list)
    location: str = ""

class CurrentJob(BaseModel):
    id: str = ""
    title: str = ""
    company: str = ""
    location: str = ""
    skills_required: List[str] = Field(default_factory=list)
    min_experience: int = 0

class ResumeState(BaseModel):
    resume_score: float = Field(default=0.0, ge=0.0, le=1.0)
    optimized_skills: List[str] = Field(default_factory=list)

class History(BaseModel):
    applied_jobs: List[str] = Field(default_factory=list)
    skipped_jobs: List[str] = Field(default_factory=list)
    generated_cover_letters: int = 0
    generated_interview_preps: int = 0

class ActionPayload(BaseModel):
    target_skills: Optional[List[str]] = None
    job_id: Optional[str] = None
    confidence: Optional[float] = None
    reason: Optional[str] = None
    cover_letter_text: Optional[str] = None
    interview_qa_pairs: Optional[List[Dict[str, str]]] = None
    raw_resume: Optional[str] = None

class JaoeAction(Action):
    """Action for the JAOE environment."""
    action_type: str = Field(..., description="OPTIMIZE_RESUME | GENERATE_COVER_LETTER | PREP_INTERVIEW | APPLY | SKIP")
    payload: ActionPayload = Field(default_factory=ActionPayload)

class JaoeObservation(Observation):
    """Observation from the JAOE environment."""
    step_count: int = 0
    max_steps: int = 15
    user_profile: UserProfile = Field(default_factory=UserProfile)
    current_job: CurrentJob = Field(default_factory=CurrentJob)
    resume_state: ResumeState = Field(default_factory=ResumeState)
    history: History = Field(default_factory=History)
