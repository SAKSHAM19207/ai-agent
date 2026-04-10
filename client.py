from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

try:
    from .models import JaoeAction, JaoeObservation, ActionPayload, UserProfile, CurrentJob, ResumeState, History
except ImportError:
    from models import JaoeAction, JaoeObservation, ActionPayload, UserProfile, CurrentJob, ResumeState, History

class JaoeEnv(
    EnvClient[JaoeAction, JaoeObservation, State]
):
    def _step_payload(self, action: JaoeAction) -> Dict:
        return {
            "action_type": action.action_type,
            "payload": action.payload.model_dump() if action.payload else {}
        }

    def _parse_result(self, payload: Dict) -> StepResult[JaoeObservation]:
        obs_data = payload.get("observation", {})
        
        user_prof = obs_data.get("user_profile", {})
        cur_job = obs_data.get("current_job", {})
        res_state = obs_data.get("resume_state", {})
        hist = obs_data.get("history", {})
        
        observation = JaoeObservation(
            step_count=obs_data.get("step_count", 0),
            max_steps=obs_data.get("max_steps", 10),
            user_profile=UserProfile(**user_prof),
            current_job=CurrentJob(**cur_job),
            resume_state=ResumeState(**res_state),
            history=History(**hist),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
