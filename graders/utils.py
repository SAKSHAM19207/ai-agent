from __future__ import annotations
from collections.abc import Iterable
from typing import Any

def clamp(score: float) -> float:
    return max(0.05, min(0.95, score))

def normalize_trajectory(trajectory: Any) -> list[dict[str, Any]]:
    if trajectory is None:
        return []

    if isinstance(trajectory, dict):
        for key in ("trajectory", "steps", "history", "records"):
            value = trajectory.get(key)
            if isinstance(value, list):
                trajectory = value
                break
        else:
            trajectory = [trajectory]
    elif hasattr(trajectory, "trajectory"):
        trajectory = getattr(trajectory, "trajectory")
    elif hasattr(trajectory, "steps"):
        trajectory = getattr(trajectory, "steps")
    elif not isinstance(trajectory, list):
        if isinstance(trajectory, Iterable) and not isinstance(trajectory, (str, bytes)):
            trajectory = list(trajectory)
        else:
            trajectory = [trajectory]

    normalized: list[dict[str, Any]] = []
    for step in trajectory:
        if isinstance(step, dict):
            normalized.append(step)
            continue

        normalized.append(
            {
                "action": getattr(step, "action", getattr(step, "action_type", None)),
                "match_ratio": getattr(step, "match_ratio", None),
                "resume_score": getattr(step, "resume_score", None),
            }
        )

    return normalized

def get_action(step: dict[str, Any]) -> str:
    action = step.get("action") or step.get("action_type")
    if isinstance(action, dict):
        action = action.get("action_type")
    return str(action or "").upper()

def get_match_ratio(step: dict[str, Any]) -> float | None:
    for key in ("match_ratio", "match", "score"):
        value = step.get(key)
        if isinstance(value, (int, float)):
            return float(value)

    observation = step.get("observation") or {}
    current_job = observation.get("current_job") or {}
    user_profile = observation.get("user_profile") or {}
    resume_state = observation.get("resume_state") or {}

    job_skills = current_job.get("skills_required") or []
    user_skills = (user_profile.get("skills") or []) + (resume_state.get("optimized_skills") or [])
    if not job_skills:
        return None

    return len(set(user_skills).intersection(job_skills)) / max(len(job_skills), 1)
