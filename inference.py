import asyncio
import os
import json
from typing import List

from openai import OpenAI

from client import JaoeEnv
from models import JaoeAction, ActionPayload

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.getenv("HF_TOKEN")

BENCHMARK = os.getenv("JAOE_BENCHMARK", "jaoe")
MAX_STEPS = 10
SUCCESS_SCORE_THRESHOLD = 0.4


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: str) -> None:
    done_val = str(done).lower()
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


def get_model_action(client: OpenAI, obs_data: dict) -> tuple[JaoeAction, str]:
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "user", "content": json.dumps(obs_data)}
            ],
            temperature=0.0,
            max_tokens=100,
            response_format={"type": "json_object"},
        )
        text = (completion.choices[0].message.content or "").strip()
        data = json.loads(text)

        action_type = data.get("action_type", "SKIP")
        payload = data.get("payload", {})

        return JaoeAction(
            action_type=action_type,
            payload=ActionPayload(**payload)
        ), text.replace("\n", "")

    except Exception:
        return JaoeAction(
            action_type="SKIP",
            payload=ActionPayload()
        ), '{"action_type":"SKIP"}'


async def run_task(task_name: str, client: OpenAI):
    # ✅ CRITICAL FIX: pass task_name
    env = JaoeEnv(base_url="http://localhost:8000", task=task_name)

    rewards: List[float] = []
    steps_taken = 0
    success = False
    score = 0.0

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
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

            log_step(
                step=step,
                action=action_str,
                reward=reward,
                done=done,
                error=error,
            )

            if done:
                break

        score = sum(rewards) / max(1.0, float(len(rewards)))
        score = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        try:
            await env.close()
        except Exception:
            pass

        log_end(
            success=success,
            steps=steps_taken,
            score=score,
            rewards=rewards,
        )


async def main() -> None:
    client = OpenAI(
        base_url=API_BASE_URL,
        api_key=HF_TOKEN or "dummy-key"
    )

    # ✅ MUST run all 3 tasks
    tasks = [
        "jcoe-easy-v0",
        "jcoe-medium-v0",
        "jcoe-hard-v0"
    ]

    for task in tasks:
        await run_task(task, client)


if __name__ == "__main__":
    asyncio.run(main())
