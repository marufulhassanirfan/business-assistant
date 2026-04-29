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

TZ           = pytz.timezone("Asia/Dhaka")
TELEGRAM_API = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"

SYSTEM_PROMPT = """You are a highly intelligent Business Assistant.
You receive input (text or audio) in English, Bangla, or mixed Bengali.
Determine the user intent and extract data.

Always respond in strict JSON only — no extra text, no markdown:
{
  "action": "REMINDER" | "MESSAGE" | "SCRAPE" | "EXPENSE" | "NOTE" | "UNKNOWN",
  "data": {},
  "confirmation_text": "Reply in the same language the user spoke."
}

Schemas:
- REMINDER : {"task_name": "str", "time_iso": "YYYY-MM-DDTHH:MM:SS+06:00"}
- MESSAGE  : {"target_name": "str", "message_content": "str"}
- SCRAPE   : {"url": "str"}
- EXPENSE  : {"category": "str", "amount": 0, "description": "str"}
- NOTE     : {"content": "str"}

Current Asia/Dhaka time: """


# ── Telegram helpers ──────────────────────────────────────────────────────────
def tg_send(chat_id, text):
    try:
        requests.post(f"{TELEGRAM_API}/sendMessage",
                      json={"chat_id": chat_id, "text": text}, timeout=30)
    except Exception as e:
        logger.error(f"tg_send failed: {e}")

def tg_typing(chat_id):
    try:
        requests.post(f"{TELEGRAM_API}/sendChatAction",
                      json={"chat_id": chat_id, "action": "typing"}, timeout=10)
    except Exception:
        pass

def tg_get_file_url(file_id):
    try:
        r = requests.post(f"{TELEGRAM_API}/getFile",
                          json={"file_id": file_id}, timeout=20)
        fpath = r.json().get("result", {}).get("file_path", "")
        if fpath:
            return f"https://api.telegram.org/file/bot{TELEGRAM_TOKEN}/{fpath}"
    except Exception as e:
        logger.error(f"getFile failed: {e}")
    return None


# ── AI Handler ────────────────────────────────────────────────────────────────
class AIHandler:
    def __init__(self, keys_str):
        self.keys   = [k.strip() for k in keys_str.split(",") if k.strip()]
        self.idx    = 0
        self.models = {
            "primary" : "gemini-3-flash-preview",
            "fallback": "gemini-2.5-flash",
        }
        if not self.keys:
            logger.warning("No GEMINI_KEYS provided.")

    def _client(self):
        return genai.Client(api_key=self.keys[self.idx])

    def _rotate(self):
        self.idx = (self.idx + 1) % max(len(self.keys), 1)
        logger.info(f"Rotated to key index {self.idx}")

    def process(self, text=None, audio_url=None, audio_bytes=None, retries=0):
        if not self.keys:
            return {"action": "UNKNOWN", "confirmation_text": "AI not configured. Add GEMINI_KEYS."}
        if retries > len(self.keys) * 2:
            return {"action": "UNKNOWN", "confirmation_text": "AI unavailable. Check Render logs."}

        model = self.models["primary"] if retries < len(self.keys) else self.models["fallback"]
        prompt = SYSTEM_PROMPT + datetime.now(TZ).isoformat()
        if text:
            prompt += f"\nUser message: {text}"

        try:
            parts = []
            if audio_bytes:
                parts.append(types.Part.from_bytes(data=audio_bytes, mime_type="audio/ogg"))
            elif audio_url:
                parts.append(types.Part.from_uri(file_uri=audio_url, mime_type="audio/ogg"))
            parts.append(types.Part.from_text(text=prompt))

            resp = self._client().models.generate_content(
                model=model,
                contents=parts,
                config=types.GenerateContentConfig(response_mime_type="application/json")
            )
            raw = resp.text.strip()
            logger.info(f"Gemini OK [{model}]: {raw[:120]}")
            return json.loads(raw)

        except Exception as e:
            logger.error(f"Gemini error (key={self.idx}, model={model}): {type(e).__name__}: {e}")
            self._rotate()
            return self.process(text=text, audio_url=audio_url,
                                audio_bytes=audio_bytes, retries=retries + 1)


# ── Google Workspace ──────────────────────────────────────────────────────────
class GoogleWorkspace:
    def __init__(self, sa_json_str, spreadsheet_id, doc_id):
        self.spreadsheet_id = spreadsheet_id
        self.doc_id         = doc_id
        self.gc             = None
        self.docs_svc       = None
        if not sa_json_str:
            logger.warning("No GOOGLE_SERVICE_ACCOUNT_JSON."); return
        try:
            creds = Credentials.from_service_account_info(
                json.loads(sa_json_str),
                scopes=["https://www.googleapis.com/auth/spreadsheets",
                        "https://www.googleapis.com/auth/documents"])
            self.gc       = gspread.authorize(creds)
            self.docs_svc = build("docs", "v1", credentials=creds)
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
        if not self.docs_svc or not self.doc_id: return False
        try:
            self.docs_svc.documents().batchUpdate(documentId=self.doc_id, body={
                "requests": [{"insertText": {"location": {"index": 1},
                    "text": f"[{datetime.now(TZ).strftime('%Y-%m-%d %H:%M:%S')}]\n{text}\n\n"}}]
            }).execute()
            return True
        except Exception as e:
            logger.error(f"append_doc_note: {e}"); return False


# ── Scraper ───────────────────────────────────────────────────────────────────
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


# ── Globals ───────────────────────────────────────────────────────────────────
ai        = AIHandler(GEMINI_KEYS_STR)
workspace = GoogleWorkspace(GOOGLE_SERVICE_ACCOUNT_JSON, SPREADSHEET_ID, GOOGLE_DOC_ID)
scheduler = BackgroundScheduler(timezone=TZ)
app       = Flask(__name__)


# ── Scheduler ─────────────────────────────────────────────────────────────────
def task_reminder_job(task_id, task_name):
    tg_send(MY_CHAT_ID, f"🔔 Reminder: {task_name}")
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
                tg_send(MY_CHAT_ID, f"🔔 Missed reminder: {vals[1]}")
                workspace.update_task_status(str(vals[0]), "Completed")
        except Exception as e:
            logger.error(f"load task: {e}")


# ── Webhook handler ───────────────────────────────────────────────────────────
def handle_update(update):
    try:
        msg = update.get("message", {})
        if not msg:
            return

        chat_id = str(msg.get("chat", {}).get("id", ""))
        if chat_id != str(MY_CHAT_ID):
            tg_send(chat_id, "Unauthorized."); return

        tg_typing(chat_id)

        text_input  = None
        audio_url   = None
        audio_bytes = None

        if "voice" in msg or "audio" in msg:
            fid = msg.get("voice", msg.get("audio", {})).get("file_id")
            if fid:
                url = tg_get_file_url(fid)
                if url:
                    try:
                        audio_bytes = requests.get(url, timeout=30).content
                    except Exception as e:
                        logger.warning(f"Audio bytes download failed: {e}")
                        audio_url = url  # fall back to URL for Gemini to fetch
                else:
                    tg_send(chat_id, "Could not get audio file. Please send text."); return
        elif "text" in msg:
            text_input = str(msg["text"])
        else:
            tg_send(chat_id, "Please send a text or voice message."); return

        result = ai.process(text=text_input, audio_url=audio_url, audio_bytes=audio_bytes)
        action = result.get("action", "UNKNOWN")
        data   = result.get("data", {})
        conf   = result.get("confirmation_text", "Done.")

        if action == "REMINDER":
            tid = workspace.append_task(data.get("task_name", ""), data.get("time_iso", ""))
            if tid:
                try:
                    scheduler.add_job(task_reminder_job, "date",
                        run_date=datetime.fromisoformat(data["time_iso"]),
                        args=[tid, data["task_name"]])
                except Exception as e:
                    logger.error(f"Schedule job: {e}")
        elif action == "EXPENSE":
            workspace.append_expense(
                data.get("category", "Misc"), data.get("amount", 0), data.get("description", ""))
        elif action == "SCRAPE":
            if data.get("url"):
                sd = scrape_url(data["url"])
                workspace.append_lead(data["url"], sd["company"], sd["contact"], sd["details"])
        elif action == "NOTE":
            workspace.append_doc_note(data.get("content", ""))

        tg_send(chat_id, conf)

    except Exception as e:
        logger.error(f"handle_update error: {e}", exc_info=True)


@app.route(f"/{TELEGRAM_TOKEN}", methods=["POST"])
def webhook():
    update = flask_request.get_json(force=True, silent=True) or {}
    logger.info(f"Update: {str(update)[:150]}")
    threading.Thread(target=handle_update, args=(update,), daemon=True).start()
    return "OK", 200

@app.route("/", methods=["GET"])
def health():
    return "Business Assistant is alive!", 200


if __name__ == "__main__":
    logger.info("Starting Business Assistant...")
    load_tasks_on_startup()
    scheduler.start()
    logger.info("Register webhook from your local PC using register_webhook.py")
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 7860)))
