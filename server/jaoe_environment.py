import os
from uuid import uuid4
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

GLOBAL_DASHBOARD_STATE = {
    "applied": 0,
    "skipped": 0,
    "cover_letters": 0,
    "resume_score": 0.51,
    "jobs": [],
    "logs": ["[SYS] Environment Initialized"]
}

try:
    from ..models import JaoeAction, JaoeObservation, UserProfile, CurrentJob, ResumeState, History
except ImportError:
    from models import JaoeAction, JaoeObservation, UserProfile, CurrentJob, ResumeState, History
class JaoeGrader:
    """
    Grader for JAOE environment that ensures the score is always strictly between 0 and 1.
    """
    def grade(self, observation) -> float:
        # Get reward from observation and clamp it to (0, 1)
        reward = getattr(observation, "reward", 0.5)
        # validator requires strictly between 0 and 1
        return float(max(0.01, min(0.99, reward)))

def get_tasks_data():
    base_user = UserProfile(
        skills=["Python", "React", "FastAPI", "SQL", "Docker"],
        experience=4,
        preferred_roles=["Full Stack Engineer", "Backend Engineer"],
        location="Remote"
    )
    
    # Task 1 (Easy): Exact matches
    easy_jobs = [
        CurrentJob(id="job_easy_1", title="Python Backend", company="Tech A", location="Remote", skills_required=["Python", "FastAPI", "Docker"], min_experience=2),
        CurrentJob(id="job_easy_2", title="Full Stack Python", company="Tech A", location="Remote", skills_required=["Python", "React", "SQL"], min_experience=3),
    ]
    
    # Task 2 (Medium): Requires optimization
    medium_jobs = [
        CurrentJob(id="job_med_1", title="Senior Python", company="Corp B", location="Remote", skills_required=["Python", "Django", "FastAPI"], min_experience=3),
        CurrentJob(id="job_med_2", title="Lead Engineer", company="Corp B", location="Remote", skills_required=["Python", "React", "Docker", "Kubernetes"], min_experience=4),
    ]
    
    # Task 3 (Hard): Mostly irrelevant
    hard_jobs = [
         CurrentJob(id="job_hard_1", title="Java Dev", company="Bank", location="On-site", skills_required=["Java", "Spring"], min_experience=5),
         CurrentJob(id="job_hard_2", title="Cyber Sec", company="Agency", location="Remote", skills_required=["Cybersecurity", "Network"], min_experience=2),
         CurrentJob(id="job_hard_3", title="Backend Engineer", company="AI Labs", location="Remote", skills_required=["Python", "FastAPI", "Redis", "Docker"], min_experience=3),
    ]
    
    return {
        "jcoe-easy-v0": {"user": base_user, "jobs": easy_jobs, "grader": JaoeGrader()},
        "jcoe-medium-v0": {"user": base_user, "jobs": medium_jobs, "grader": JaoeGrader()},
        "jcoe-hard-v0": {"user": base_user, "jobs": hard_jobs, "grader": JaoeGrader()}
    }

class JaoeEnvironment(Environment):
    """
    Job Application Optimization Environment (JAOE).
    Automated evaluation of job filtering, optimization, and application behavior.
    """
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self.tasks_data = get_tasks_data()
        # Initialize default task
        self.task_id = "jcoe-easy-v0"

    def get_grader(self):
        return JaoeGrader()

    def reset(self) -> JaoeObservation:
        self._state = State(episode_id=str(uuid4()), step_count=0)
        
        GLOBAL_DASHBOARD_STATE["applied"] = 0
        GLOBAL_DASHBOARD_STATE["skipped"] = 0
        GLOBAL_DASHBOARD_STATE["cover_letters"] = 0
        GLOBAL_DASHBOARD_STATE["resume_score"] = 0.5
        GLOBAL_DASHBOARD_STATE["jobs"] = []
        GLOBAL_DASHBOARD_STATE["logs"] = ["[SYS] Environment Reset"]
        
        # Read task from ENV or default to Easy
        self.task_id = os.getenv("JAOE_TASK", "jcoe-easy-v0")
        if self.task_id not in self.tasks_data:
            self.task_id = "jcoe-easy-v0"
            
        task_info = self.tasks_data[self.task_id]
        self.user = task_info["user"]
        self.jobs = task_info["jobs"]
        
        self.max_steps = 10
        self.current_job_idx = 0
        self.resume_score = 0.5
        self.optimized_skills = []
        self.applied_jobs = []
        self.skipped_jobs = []
        self.generated_cover_letters = 0
        self.generated_interview_preps = 0
        self.total_reward = 0.05

        return self._get_obs(reward=self.total_reward, done=False)

    def _get_obs(self, reward: float, done: bool) -> JaoeObservation:
        if self.current_job_idx < len(self.jobs):
            current_job = self.jobs[self.current_job_idx]
        else:
            current_job = CurrentJob(id="done", title="Done", company="Done", location="None", skills_required=[], min_experience=0)
            
        obs = JaoeObservation(
            step_count=self._state.step_count,
            max_steps=self.max_steps,
            user_profile=self.user,
            current_job=current_job,
            resume_state=ResumeState(resume_score=self.resume_score, optimized_skills=self.optimized_skills),
            history=History(
                applied_jobs=self.applied_jobs, 
                skipped_jobs=self.skipped_jobs,
                generated_cover_letters=self.generated_cover_letters,
                generated_interview_preps=self.generated_interview_preps
            ),
            reward=reward,
            done=done
        )
        # Required OpenEnv metadata
        obs.metadata = {"task": self.task_id}
        return obs

    def step(self, action: JaoeAction) -> JaoeObservation:
        self._state.step_count += 1
        
        if self.current_job_idx >= len(self.jobs) or self._state.step_count > self.max_steps:
            return self._get_obs(reward=0.01, done=True)
            
        current_job = self.jobs[self.current_job_idx]
        user_skills = set(self.user.skills + self.optimized_skills)
        job_skills = set(current_job.skills_required)
        
        match_ratio = len(user_skills.intersection(job_skills)) / max(len(job_skills), 1)
        
        step_reward = 0.0
        msg = ""
        act_type = action.action_type
        
        if act_type == "SKIP":
            if match_ratio < 0.5:
                step_reward += 0.3
                msg = "Good skip."
            else:
                step_reward -= 0.3
                msg = "Penalty: skipped high match."
            self.skipped_jobs.append(current_job.id)
            self.current_job_idx += 1
            self.resume_score = 0.5
            self.optimized_skills = []
            
        elif act_type == "OPTIMIZE_RESUME":
            targets = action.payload.target_skills or []
            valid_targets = [s for s in targets if s in job_skills]
            if len(valid_targets) > 0 and self.resume_score < 0.7:
                self.resume_score = min(0.98, self.resume_score + 0.3)
                for tgt in valid_targets:
                    if tgt not in self.optimized_skills:
                        self.optimized_skills.append(tgt)
                step_reward += 0.2
                msg = "Optimized resume."
            else:
                step_reward -= 0.2
                msg = "Inefficient optimize."
                
        elif act_type == "GENERATE_COVER_LETTER":
            cl_text = action.payload.cover_letter_text or ""
            # Heuristic check for length and company mention
            if len(cl_text.split()) > 20 and current_job.company.lower() in cl_text.lower():
                step_reward += 0.2
                self.generated_cover_letters += 1
                msg = "Cover letter generated successfully."
            else:
                step_reward -= 0.1
                msg = "Cover letter failed heuristic checks."
                
        elif act_type == "PREP_INTERVIEW":
            qa_pairs = action.payload.interview_qa_pairs or []
            if len(qa_pairs) >= 2:
                step_reward += 0.2
                self.generated_interview_preps += 1
                msg = "Interview prep successfully generated."
            else:
                step_reward -= 0.1
                msg = "Insufficient interview questions."

        elif act_type == "APPLY":
            if match_ratio >= 0.7 or (match_ratio >= 0.5 and self.resume_score >= 0.7):
                step_reward += 0.5
                msg = "Good apply."
            else:
                step_reward -= 0.5
                msg = "Bad apply."
            self.applied_jobs.append(current_job.id)
            self.current_job_idx += 1
            self.resume_score = 0.5
            self.optimized_skills = []
        else:
            step_reward -= 0.1
            msg = "Invalid action."

        done = self.current_job_idx >= len(self.jobs) or self._state.step_count >= self.max_steps
        if done and self._state.step_count <= self.max_steps:
            step_reward += 0.2 # Completion bonus
            
        self.total_reward = max(0.01, min(0.99, self.total_reward + step_reward))
            
        GLOBAL_DASHBOARD_STATE["resume_score"] = self.resume_score
        GLOBAL_DASHBOARD_STATE["applied"] = len(self.applied_jobs)
        GLOBAL_DASHBOARD_STATE["skipped"] = len(self.skipped_jobs)
        GLOBAL_DASHBOARD_STATE["cover_letters"] = self.generated_cover_letters
        GLOBAL_DASHBOARD_STATE["logs"].insert(0, f"[AGENT] Step {self._state.step_count}: {msg}")
        GLOBAL_DASHBOARD_STATE["jobs"].insert(0, {
            "title": current_job.title,
            "company": current_job.company,
            "action": act_type
        })
        # Keep logs manageable
        if len(GLOBAL_DASHBOARD_STATE["logs"]) > 50: GLOBAL_DASHBOARD_STATE["logs"].pop()
        if len(GLOBAL_DASHBOARD_STATE["jobs"]) > 50: GLOBAL_DASHBOARD_STATE["jobs"].pop()

        obs = self._get_obs(reward=self.total_reward, done=done)
        obs.metadata["msg"] = msg
        return obs

    @property
    def state(self) -> State:
        return self._state
