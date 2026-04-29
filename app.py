import os
import json
import logging
import time
import threading
from datetime import datetime
import pytz
import requests
from flask import Flask, request as flask_request, jsonify
from bs4 import BeautifulSoup
from google import genai
from google.genai import types
import gspread
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from apscheduler.schedulers.background import BackgroundScheduler

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

GEMINI_KEYS_STR             = os.environ.get("GEMINI_KEYS", "")
TELEGRAM_TOKEN              = os.environ.get("TELEGRAM_TOKEN", "")
MY_CHAT_ID                  = os.environ.get("MY_CHAT_ID", "")
GOOGLE_SERVICE_ACCOUNT_JSON = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON", "")
SPREADSHEET_ID              = os.environ.get("SPREADSHEET_ID", "")
GOOGLE_DOC_ID               = os.environ.get("GOOGLE_DOC_ID", "")

TZ = pytz.timezone("Asia/Dhaka")

SYSTEM_PROMPT = """You are a highly intelligent Business Assistant.
You receive input (text or audio) in English, Bangla, or mixed Bengali.
Determine the user intent and extract data.

Always respond in strict JSON:
{
  "action": "REMINDER" | "MESSAGE" | "SCRAPE" | "EXPENSE" | "NOTE" | "UNKNOWN",
  "data": { ... },
  "confirmation_text": "Reply in the same language the user spoke."
}

Schemas:
- REMINDER : {"task_name": "str", "time_iso": "YYYY-MM-DDTHH:MM:SS+06:00"}
- MESSAGE  : {"target_name": "str", "message_content": "str"}
- SCRAPE   : {"url": "str"}
- EXPENSE  : {"category": "str", "amount": 0, "description": "str"}
- NOTE     : {"content": "str"}

Current Asia/Dhaka time: """


class AIHandler:
    def __init__(self, keys_str):
        self.keys   = [k.strip() for k in keys_str.split(",") if k.strip()]
        self.idx    = 0
        self.models = {
            "primary" : "gemini-3-flash-preview",
            "fallback": "gemini-2.5-flash",
        }

    def _client(self):
        return genai.Client(api_key=self.keys[self.idx])

    def _rotate(self):
        self.idx = (self.idx + 1) % max(len(self.keys), 1)

    def process_input(self, audio_bytes=None, audio_url=None, text=None, retries=0):
        if not self.keys:
            return {"action": "UNKNOWN", "confirmation_text": "AI not configured. Add GEMINI_KEYS secret."}
        if retries > len(self.keys) * 2:
            return {"action": "UNKNOWN", "confirmation_text": f"All AI services unavailable. Last error logged above."}
        tier = "primary" if retries < len(self.keys) else "fallback"
        model = self.models[tier]
        try:
            client = genai.Client(api_key=self.keys[self.idx])
            prompt = SYSTEM_PROMPT + datetime.now(TZ).isoformat()
            if text:
                prompt += "\nUser said: " + text

            contents = prompt
            if audio_bytes:
                contents = [
                    types.Part.from_bytes(data=audio_bytes, mime_type="audio/ogg"),
                    types.Part.from_text(text=prompt)
                ]
            elif audio_url:
                contents = [
                    types.Part.from_uri(file_uri=audio_url, mime_type="audio/ogg"),
                    types.Part.from_text(text=prompt)
                ]

            resp = client.models.generate_content(
                model=model,
                contents=contents,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json"
                )
            )
            raw = resp.text.strip()
            logger.info(f"Gemini raw response: {raw[:200]}")
            return json.loads(raw)
        except Exception as e:
            logger.error(f"Gemini error (key={self.idx}, model={model}): {type(e).__name__}: {e}")
            self._rotate()
            return self.process_input(audio_bytes, text, retries + 1)



class GoogleWorkspace:
    def __init__(self, sa_json_str, spreadsheet_id, doc_id):
        self.spreadsheet_id = spreadsheet_id
        self.doc_id = doc_id
        self.gc = None
        self.docs_service = None
        if not sa_json_str:
            return
        try:
            creds = Credentials.from_service_account_info(
                json.loads(sa_json_str),
                scopes=["https://www.googleapis.com/auth/spreadsheets",
                        "https://www.googleapis.com/auth/documents"])
            self.gc = gspread.authorize(creds)
            self.docs_service = build("docs", "v1", credentials=creds)
            logger.info("Google Workspace connected.")
        except Exception as e:
            logger.error(f"Workspace init: {e}")

    def _sheet(self, tab):
        return self.gc.open_by_key(self.spreadsheet_id).worksheet(tab)

    def append_task(self, task_name, time_iso):
        if not self.gc: return False
        try:
            tid = str(int(time.time()))
            self._sheet("Tasks").append_row([tid, task_name, time_iso, "Pending"])
            return tid
        except Exception as e:
            logger.error(f"append_task: {e}"); return False

    def update_task_status(self, task_id, status):
        if not self.gc: return
        try:
            cells = self._sheet("Tasks").findall(str(task_id))
            if cells:
                self._sheet("Tasks").update_cell(cells[0].row, 4, status)
        except Exception as e:
            logger.error(f"update_task_status: {e}")

    def get_pending_tasks(self):
        if not self.gc: return []
        try:
            return [r for r in self._sheet("Tasks").get_all_records()
                    if list(r.values())[3] == "Pending"]
        except Exception as e:
            logger.error(f"get_pending_tasks: {e}"); return []

    def append_expense(self, category, amount, description):
        if not self.gc: return False
        try:
            self._sheet("Expenses").append_row(
                [datetime.now(TZ).strftime("%Y-%m-%d"), category, amount, description])
            return True
        except Exception as e:
            logger.error(f"append_expense: {e}"); return False

    def append_lead(self, url, company, contact, details):
        if not self.gc: return False
        try:
            self._sheet("Leads").append_row(
                [datetime.now(TZ).strftime("%Y-%m-%d"), url, company, contact, details])
            return True
        except Exception as e:
            logger.error(f"append_lead: {e}"); return False

    def append_doc_note(self, text):
        if not self.docs_service or not self.doc_id: return False
        try:
            self.docs_service.documents().batchUpdate(documentId=self.doc_id, body={
                "requests": [{"insertText": {"location": {"index": 1},
                    "text": f"[{datetime.now(TZ).strftime('%Y-%m-%d %H:%M:%S')}]\n{text}\n\n"}}]
            }).execute()
            return True
        except Exception as e:
            logger.error(f"append_doc_note: {e}"); return False


def scrape_url(url):
    try:
        r    = requests.get(url, timeout=15)
        soup = BeautifulSoup(r.content, "html.parser")
        title  = soup.title.string.strip() if soup.title else "Unknown"
        emails = [a["href"].replace("mailto:", "") for a in soup.find_all("a", href=True)
                  if a["href"].startswith("mailto:")]
        return {"company": title, "contact": emails[0] if emails else "Not found", "details": "OK"}
    except Exception as e:
        return {"company": "Error", "contact": "Error", "details": str(e)}


ai        = AIHandler(GEMINI_KEYS_STR)
workspace = GoogleWorkspace(GOOGLE_SERVICE_ACCOUNT_JSON, SPREADSHEET_ID, GOOGLE_DOC_ID)
scheduler = BackgroundScheduler(timezone=TZ)
app       = Flask(__name__)

# Pending reminder queue — scheduler stores messages here, next webhook call flushes them
pending_messages = []
pending_lock     = threading.Lock()


def queue_message(text):
    """Store a message to be sent on next webhook response (avoids outbound call)."""
    with pending_lock:
        pending_messages.append(text)


def task_reminder_job(task_id, task_name):
    queue_message(f"🔔 Reminder: {task_name}")
    workspace.update_task_status(task_id, "Completed")


def load_tasks_on_startup():
    for task in workspace.get_pending_tasks():
        try:
            vals      = list(task.values())
            task_time = datetime.fromisoformat(str(vals[2]))
            if task_time > datetime.now(TZ):
                scheduler.add_job(task_reminder_job, "date",
                                  run_date=task_time, args=[str(vals[0]), str(vals[1])])
                logger.info(f"Reloaded: {vals[1]} @ {task_time}")
            else:
                queue_message(f"🔔 Missed reminder: {vals[1]}")
                workspace.update_task_status(str(vals[0]), "Completed")
        except Exception as e:
            logger.error(f"load task: {e}")


def tg_reply(chat_id, text):
    """Build a Telegram sendMessage response — returned directly in HTTP body."""
    return jsonify({"method": "sendMessage", "chat_id": chat_id, "text": text})


def process_update(update):
    """Process one Telegram update and return (chat_id, reply_text)."""
    msg = update.get("message", {})
    if not msg:
        return None, None

    chat_id = str(msg.get("chat", {}).get("id", ""))
    if chat_id != str(MY_CHAT_ID):
        return chat_id, "Unauthorized."

    audio_bytes, audio_url, text_input = None, None, None

    if "voice" in msg or "audio" in msg:
        fid = msg.get("voice", msg.get("audio", {})).get("file_id")
        if fid:
            try:
                # Get the file path from Telegram (short quick call, not long-polling)
                api_url   = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/getFile"
                file_resp = requests.post(api_url, json={"file_id": fid}, timeout=20)
                fpath     = file_resp.json().get("result", {}).get("file_path", "")
                if fpath:
                    # Pass the URL directly to Gemini — Gemini fetches it, not HF
                    tg_file_url = f"https://api.telegram.org/file/bot{TELEGRAM_TOKEN}/{fpath}"
                    audio_url   = tg_file_url
                    text_input  = None
                else:
                    return chat_id, "Could not get audio file info from Telegram."
            except Exception as e:
                logger.error(f"getFile failed: {e}")
                return chat_id, "Could not reach Telegram to get audio info. Please send text."
    elif "text" in msg:
        text_input = msg["text"]
        audio_url  = None
    else:
        return chat_id, "Please send a text or voice message."

    result = ai.process_input(audio_bytes=audio_bytes, audio_url=audio_url, text=text_input)
    action = result.get("action", "UNKNOWN")
    data   = result.get("data", {})
    conf   = result.get("confirmation_text", "Done.")

    try:
        if action == "REMINDER":
            tid = workspace.append_task(data.get("task_name"), data.get("time_iso"))
            if tid:
                scheduler.add_job(task_reminder_job, "date",
                    run_date=datetime.fromisoformat(data["time_iso"]),
                    args=[tid, data["task_name"]])
        elif action == "EXPENSE":
            workspace.append_expense(
                data.get("category", "Misc"), data.get("amount", 0), data.get("description", ""))
        elif action == "SCRAPE":
            if data.get("url"):
                sd = scrape_url(data["url"])
                workspace.append_lead(data["url"], sd["company"], sd["contact"], sd["details"])
        elif action == "NOTE":
            workspace.append_doc_note(data.get("content", ""))
    except Exception as e:
        logger.error(f"Action {action}: {e}")
        conf += f" (error: {e})"

    return chat_id, conf


@app.route(f"/{TELEGRAM_TOKEN}", methods=["POST"])
def webhook():
    update = flask_request.get_json(force=True)
    logger.info(f"Update received: {json.dumps(update)[:200]}")

    # Flush any queued reminder messages first (prepend to response)
    with pending_lock:
        queued = list(pending_messages)
        pending_messages.clear()

    chat_id, reply = process_update(update)

    # Combine queued reminders + current reply
    all_text = "\n\n".join(queued)
    if reply:
        all_text = (all_text + "\n\n" + reply).strip() if all_text else reply

    if chat_id and all_text:
        return tg_reply(chat_id, all_text)

    return "OK", 200


@app.route("/", methods=["GET"])
def health():
    return "✅ Business Assistant is alive!", 200


if __name__ == "__main__":
    logger.info("Starting Business Assistant (webhook mode)...")
    load_tasks_on_startup()
    scheduler.start()
    logger.info("Webhook receiver ready. Register webhook from your local PC using register_webhook.py")
    app.run(host="0.0.0.0", port=7860)







