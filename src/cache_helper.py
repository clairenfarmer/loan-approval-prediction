import json
import time
from pathlib import Path

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

LOG_PATH = Path("report/ai_log.json")
MODEL_NAME = "gemini-3.5-flash-lite"

llm = ChatGoogleGenerativeAI(model=MODEL_NAME)


def to_text(content):
    """Return plain text whether the model gave a string or a list of content blocks."""
    if isinstance(content, str):
        return content
    return "".join(
        block.get("text", "") if isinstance(block, dict) else str(block)
        for block in content
    )


def load_log():
    return json.loads(LOG_PATH.read_text()) if LOG_PATH.exists() else {}


def write_log(log):
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.write_text(json.dumps(log, indent=2))


def ask_ai(step_id, prompt, refresh=False):
    """Ask Gemini once per step_id, then reuse the logged answer.
    Retries with backoff on temporary errors. Failures are never cached."""
    log = load_log()
    if step_id in log and not refresh:
        return log[step_id]["response"]

    last_error = None
    for attempt in range(4):
        try:
            response = llm.invoke(prompt)
            text = to_text(response.content)
            log = load_log()
            log[step_id] = {
                "prompt": prompt,
                "response": text,
                "model": MODEL_NAME,
                "decision": None,
                "justification": "",
            }
            write_log(log)
            return text
        except Exception as e:
            last_error = e
            time.sleep(2 ** attempt)   # wait 1s, 2s, 4s, 8s between attempts
    raise RuntimeError(f"Gemini is unavailable right now: {last_error}")