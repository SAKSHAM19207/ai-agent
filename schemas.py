from pydantic import BaseModel
from typing import List, Optional, Any, Dict

class UserProfile(BaseModel):
    skills: List[str]
    experience: int
    preferred_roles: List[str]
    location: str

class CurrentJob(BaseModel):
    id: str
    title: str
    company: str
    location: str
    skills_required: List[str]
    min_experience: int

class ResumeState(BaseModel):
    resume_score: float
    optimized_skills: List[str]

class History(BaseModel):
    applied_jobs: List[str]
    skipped_jobs: List[str]

class Observation(BaseModel):
    step_count: int
    max_steps: int
    user_profile: UserProfile
    current_job: CurrentJob
    resume_state: ResumeState
    history: History

class ActionPayload(BaseModel):
    target_skills: Optional[List[str]] = None
    job_id: Optional[str] = None
    confidence: Optional[float] = None
    reason: Optional[str] = None

class Action(BaseModel):
    action_type: str  # "OPTIMIZE_RESUME" | "APPLY" | "SKIP"
    payload: ActionPayload

class StepResponse(BaseModel):
    observation: Observation
    reward: float
    done: bool
    info: Dict[str, Any]
