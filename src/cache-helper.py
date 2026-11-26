import json
from pathlib import Path
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
LOG_PATH = Path("report/ai_log.json")

def get_llm():
    return ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)

def load_log():
    return json.loads(LOG_PATH.read_text()) if LOG_PATH.exists() else {}

def ask_ai(step_id, prompt, refresh=False):
    """Call Gemini once per step_id and cache the prompt and response."""
    log = load_log()
    if step_id in log and not refresh:
        return log[step_id]["response"]
    response = get_llm().invoke(prompt).content
    log[step_id] = {"prompt": prompt, "response": response,
                    "decision": None, "justification": ""}
    LOG_PATH.write_text(json.dumps(log, indent=2))
    return response

def save_decision(step_id, decision, justification):
    log = load_log()
    log[step_id]["decision"] = decision
    log[step_id]["justification"] = justification
    LOG_PATH.write_text(json.dumps(log, indent=2))