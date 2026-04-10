import asyncio
import os
import textwrap
import json
from typing import List, Optional

from openai import OpenAI

from client import JaoeEnv
from models import JaoeAction, ActionPayload

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.getenv("HF_TOKEN")

# Optional - if you use from_docker_image():
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
TASK_NAME = os.getenv("JAOE_TASK", "jcoe-easy-v0")
BENCHMARK = os.getenv("JAOE_BENCHMARK", "jaoe")
MAX_STEPS = 10
SUCCESS_SCORE_THRESHOLD = 0.4

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an AI Job Application Agent operating in the JAOE.
    Evaluate match ratio: |user.skills ∩ job.skills_required| / max(1, |job.skills_required|).
    Decision logic:
    If match < 0.5 -> SKIP
    If 0.5 <= match < 0.7 AND resume_score < 0.7 -> action_type: OPTIMIZE_RESUME
    If match >= 0.7 -> Provide PREP_INTERVIEW or GENERATE_COVER_LETTER before executing APPLY.
    For GENERATE_COVER_LETTER, provide 'cover_letter_text' (>20 words) in payload.
    For PREP_INTERVIEW, provide 'interview_qa_pairs' in payload.
    Ultimately, ALWAYS output EXCLUSIVELY VALID JSON:
    { "action_type": "...", "payload": { ... } }
    """
).strip()

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

def get_model_action(client: OpenAI, obs_data: dict) -> tuple[JaoeAction, str]:
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(obs_data)}
            ],
            temperature=0.0,
            max_tokens=200,
            response_format={ "type": "json_object" }
        )
        text = (completion.choices[0].message.content or "").strip()
        data = json.loads(text)
        action_type = data.get("action_type", "SKIP")
        payload = data.get("payload", {})
        
        return JaoeAction(action_type=action_type, payload=ActionPayload(**payload)), text.replace('\n', '')
    except Exception as exc:
        return JaoeAction(action_type="SKIP", payload=ActionPayload(reason="fallback")), f'{{"action_type":"SKIP", "error":"{exc}"}}'

async def wait_for_server(env: JaoeEnv, max_retries: int = 5, delay: float = 2.0) -> bool:
    """Wait for the environment server to be ready by attempting reset."""
    for i in range(max_retries):
        try:
            print(f"[INFO] Attempting to connect to environment (Attempt {i+1}/{max_retries})...", flush=True)
            await env.reset()
            print(f"[INFO] Successfully connected to environment.", flush=True)
            return True
        except Exception as e:
            print(f"[WARN] Connection attempt {i+1} failed: {e}. Retrying in {delay}s...", flush=True)
            await asyncio.sleep(delay)
    return False

async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN or "dummy-key")

    # Try ports in order: environment variable PORT, then 8000 (standard), then 7860 (Hugging Face)
    env_port = os.getenv("PORT")
    ports_to_try = [env_port] if env_port else ["8000", "7860"]
    
    env = None
    connected = False
    
    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    for port in ports_to_try:
        print(f"[INFO] Testing connection on port {port}...", flush=True)
        temp_env = JaoeEnv(base_url=f"http://localhost:{port}")
        if await wait_for_server(temp_env, max_retries=3, delay=2.0):
            env = temp_env
            connected = True
            break
    
    rewards: List[float] = []
    steps_taken = 0
    success = False
    score = 0.0

    if not connected:
        print("[ERROR] Could not connect to environment server on any tried port.", flush=True)
        log_end(success=False, steps=0, score=0.0, rewards=[])
        return

    try:
        # Initial reset successful from wait_for_server
        result = await env.reset()
        
        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break
            
            obs_json = result.observation.model_dump()
            action_obj, action_str = get_model_action(client, obs_json)

            error = None
            try:
                result = await env.step(action_obj)
                reward = result.reward or 0.0
                done = result.done
            except Exception as e:
                reward = 0.0
                done = True
                error = str(e)
            
            rewards.append(reward)
            steps_taken = step
            log_step(step=step, action=action_str, reward=reward, done=done, error=error)

            if done:
                break

        score = sum(rewards) / max(1.0, float(len(rewards)))
        score = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as e:
        print(f"[ERROR] Fatal error during inference: {e}", flush=True)
    finally:
        if env:
            try:
                await env.close()
            except:
                pass
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"[CRITICAL] Unhandled exception: {e}", flush=True)
