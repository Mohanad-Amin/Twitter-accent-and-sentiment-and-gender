import os
import re
import time
import json
import random
import requests
import pandas as pd
from google.colab import drive, userdata

# ==============================
# إعداد Google Drive
# ==============================
drive.mount('/content/drive')

# ==============================
# إعدادات OpenAI Batch API
# ==============================
OPENAI_FILES_URL = "https://api.openai.com/v1/files"
OPENAI_BATCHES_URL = "https://api.openai.com/v1/batches"
OPENAI_CHAT_ENDPOINT = "/v1/chat/completions"
MODEL = "gpt-4o-mini"
BATCH_COMPLETION_WINDOW = "24h"

# ==============================
# المفتاح
# ==============================
API_KEY = ""
try:
    API_KEY = userdata.get('Mohanad') or userdata.get('OPENAI_API_KEY') or ""
except Exception:
    pass
if not API_KEY:
    API_KEY = os.environ.get('Mohanad', '') or os.environ.get('OPENAI_API_KEY', '')
if not API_KEY:
    raise SystemExit("❌ لم يتم العثور على مفتاح OpenAI (Mohanad / OPENAI_API_KEY)")

# ==============================
# المسارات
# ==============================
INPUT_XLSX = "/content/drive/MyDrive/Tweet_Project/Data/Gender_dallah.xlsx"
REQ_JSONL  = "/content/drive/MyDrive/Tweet_Project/Data/gender_requests.jsonl"
MAP_JSONL  = "/content/drive/MyDrive/Tweet_Project/Data/gender_mapping.jsonl"
RAW_OUT    = "/content/drive/MyDrive/Tweet_Project/Data/gender_batch_output.jsonl"
OUT_XLSX   = "/content/drive/MyDrive/Tweet_Project/Data/Gender_dallah_with_genders.xlsx"

# ==============================
# تحميل البيانات (من السطر 2150 إلى 3000)
# ==============================
def load_rows_2150_to_3000(path):
    df = pd.read_excel(path)
    # توحيد الأسماء المحتملة للأعمدة
    cols = {c.strip().lower(): c for c in df.columns}
    def pick(*cands):
        for c in cands:
            if c in cols: return cols[c]
        return None

    col_name  = pick("author.name", "name", "author name", "author_name")
    col_user  = pick("author.username", "username", "user", "screen_name", "author_username")
    col_desc  = pick("author.description", "description", "bio", "author_description")

    if not (col_name and col_user and col_desc):
        raise ValueError(f"لم يتم العثور على الأعمدة المطلوبة. الأعمدة الموجودة: {list(df.columns)}")

    # أخذ الأسطر من 2149 (index) إلى 2999 (index) - أي من السطر 2150 إلى 3000
    df = df[[col_name, col_user, col_desc]].iloc[4000:6000].copy()
    df.columns = ["author.name", "author.username", "author.description"]
    # ملء فراغات
    for c in df.columns:
        df[c] = df[c].astype(str).fillna("").replace("nan", "", regex=False)

    print(f"📊 تم تحميل {len(df)} سطر (من السطر 2150 إلى {2149 + len(df)})")
    return df

# ==============================
# بناء رسالة التصنيف
# ==============================
def j(s):  # JSON-safe
    return json.dumps("" if s is None else str(s), ensure_ascii=False)

SYSTEM_PROMPT = (
    "You are a cautious classifier. Task: infer the account holder's *perceived* gender "
    "purely from provided fields (display name, username, description). "
    "Output one of: male, female, unknown.\n\n"
    "Rules:\n"
    "- If organization/brand/media/news/job board, or multiple people, return unknown.\n"
    "- If evidence is insufficient/ambiguous, return unknown.\n"
    "- Prefer not to guess. Only use clear linguistic/onomastic cues (Arabic/English names, pronouns, emojis).\n"
    "- Do not add explanations; return JSON only."
)

def build_user_message(row):
    name = j(row["author.name"])
    user = j(row["author.username"])
    desc = j(row["author.description"])

    # تعليمات صارمة لإخراج JSON فقط
    msg = (
        "Classify perceived gender for this X/Twitter account.\n"
        f"display_name: {name}\n"
        f"username: {user}\n"
        f"description: {desc}\n\n"
        "Return ONLY valid JSON object exactly like: {\"gender\":\"male|female|unknown\"}"
    )
    return msg

# ==============================
# إعداد JSONL + Mapping
# ==============================
def prepare_jsonl_for_batch(df, req_path, map_path):
    with open(req_path, "w", encoding="utf-8") as jf, open(map_path, "w", encoding="utf-8") as mf:
        ts = int(time.time() * 1000)
        for i, row in df.reset_index(drop=True).iterrows():
            custom_id = f"gender-2150-3000-{ts}-{i+1}"
            user_msg = build_user_message(row)
            body = {
                "custom_id": custom_id,
                "method": "POST",
                "url": OPENAI_CHAT_ENDPOINT,
                "body": {
                    "model": MODEL,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user",   "content": user_msg}
                    ],
                    "response_format": {"type": "json_object"},
                    "temperature": 0.0,
                    "max_tokens": 20
                }
            }
            jf.write(json.dumps(body, ensure_ascii=False) + "\n")
            mf.write(json.dumps({
                "custom_id": custom_id,
                "row_index": int(i),
                "original_row": int(i + 2150),  # إضافة رقم السطر الأصلي
                "name": row["author.name"],
                "username": row["author.username"],
                "description": row["author.description"]
            }, ensure_ascii=False) + "\n")
    print(f"🗂️ JSONL الطلبات: {req_path}")
    print(f"🗺️ JSONL المابنج: {map_path}")

# ==============================
# وظائف Batch API
# ==============================
def upload_file_for_batch(jsonl_path: str) -> str:
    headers = {"Authorization": f"Bearer {API_KEY}"}
    files = {"file": (os.path.basename(jsonl_path), open(jsonl_path, 'rb'))}
    data = {"purpose": "batch"}
    r = requests.post(OPENAI_FILES_URL, headers=headers, files=files, data=data, timeout=180)
    if r.status_code != 200:
        raise RuntimeError(f"File upload failed: {r.status_code} {r.text}")
    fid = r.json()["id"]
    print(f"📎 تم الرفع: file_id={fid}")
    return fid

def create_batch_job(file_id: str, completion_window: str = BATCH_COMPLETION_WINDOW) -> str:
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    payload = {"input_file_id": file_id, "endpoint": OPENAI_CHAT_ENDPOINT, "completion_window": completion_window}
    r = requests.post(OPENAI_BATCHES_URL, headers=headers, json=payload, timeout=60)
    if r.status_code != 200:
        raise RuntimeError(f"Create batch failed: {r.status_code} {r.text}")
    bid = r.json()["id"]
    print(f"📦 تم إنشاء Batch: batch_id={bid}")
    return bid

def poll_batch_until_done(batch_id: str, poll_interval: int = 20) -> dict:
    headers = {"Authorization": f"Bearer {API_KEY}"}
    while True:
        r = requests.get(f"{OPENAI_BATCHES_URL}/{batch_id}", headers=headers, timeout=60)
        if r.status_code != 200:
            raise RuntimeError(f"Batch status failed: {r.status_code} {r.text}")
        obj = r.json()
        status = obj.get("status")
        print(f"⏳ حالة الباتش: {status}")
        if status in ("completed", "failed", "canceled"):
            return obj
        time.sleep(poll_interval)

def download_file_content(file_id: str) -> str:
    headers = {"Authorization": f"Bearer {API_KEY}"}
    r = requests.get(f"{OPENAI_FILES_URL}/{file_id}/content", headers=headers, timeout=180)
    if r.status_code != 200:
        raise RuntimeError(f"Download file failed: {r.status_code} {r.text}")
    return r.text

# ==============================
# قراءة المابنج
# ==============================
def load_mapping(map_path: str) -> dict:
    mapping = {}
    with open(map_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            mapping[row["custom_id"]] = row
    return mapping

# ==============================
# تحليل الإخراج ودمجه
# ==============================
def parse_and_merge(output_jsonl_text: str, mapping: dict) -> pd.DataFrame:
    out_rows = []
    for line in output_jsonl_text.splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)

        if obj.get("error"):
            # سطر فشل في الباتش
            continue

        resp = obj.get("response") or {}
        if resp.get("status_code") != 200:
            continue

        body = resp.get("body") or {}
        choices = body.get("choices") or []
        if not choices:
            continue

        content = choices[0].get("message", {}).get("content", "").strip()
        try:
            content_obj = json.loads(content)
        except json.JSONDecodeError:
            continue

        pred = str(content_obj.get("gender", "")).strip().lower()
        if pred not in ("male", "female", "unknown"):
            pred = "unknown"

        cid = obj.get("custom_id")
        meta = mapping.get(cid, {})
        out_rows.append({
            "original_row": meta.get("original_row", ""),
            "author.name": meta.get("name", ""),
            "author.username": meta.get("username", ""),
            "author.description": meta.get("description", ""),
            "gender": pred
        })

    return pd.DataFrame(out_rows)

# ==============================
# حفظ إلى إكسل + إحصائيات بسيطة
# ==============================
def save_labeled_excel(df: pd.DataFrame, path: str):
    if df.empty:
        print("❌ لا توجد نتائج للحفظ.")
        return
    # إحصائيات
    counts = df["gender"].value_counts(dropna=False)
    stats_df = pd.DataFrame({
        "gender": counts.index,
        "count": counts.values,
        "percent_%": (counts.values / len(df) * 100).round(2)
    })
    try:
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="labeled")
            stats_df.to_excel(writer, index=False, sheet_name="stats")
        print(f"✅ تم حفظ النتائج في: {path}")
        print(f"📊 إجمالي الأسطر المعالجة: {len(df)} (من السطر 2150 إلى 3000)")
        print(stats_df)
    except Exception as e:
        print(f"❌ خطأ في الحفظ: {e}")

# ==============================
# Main
# ==============================
def main():
    print("🚀 Gender Batch (male/female/unknown) — الأسطر 2150-3000 — GPT-4o — OpenAI Batch API فقط")
    df = load_rows_2150_to_3000(INPUT_XLSX)

    # 1) إنشاء الطلبات + المابنج
    prepare_jsonl_for_batch(df, REQ_JSONL, MAP_JSONL)

    # 2) رفع الملف وإنشاء Batch
    file_id = upload_file_for_batch(REQ_JSONL)
    batch_id = create_batch_job(file_id, completion_window=BATCH_COMPLETION_WINDOW)

    # 3) الانتظار حتى الاكتمال
    status_obj = poll_batch_until_done(batch_id, poll_interval=20)
    final_status = status_obj.get("status")
    print(f"✅ الحالة النهائية: {final_status}")
    if final_status != "completed":
        print("❌ لم يكتمل الباتش بنجاح:", status_obj)
        return

    # 4) تنزيل النتائج الخام
    out_file_id = status_obj.get("output_file_id")
    if not out_file_id:
        print("❌ لا يوجد output_file_id.")
        return
    out_text = download_file_content(out_file_id)
    with open(RAW_OUT, "w", encoding="utf-8") as f:
        f.write(out_text)
    print(f"📥 تم حفظ نتائج الباتش الخام: {RAW_OUT}")

    # 5) دمج النتائج مع الأصل
    mapping = load_mapping(MAP_JSONL)
    labeled_df = parse_and_merge(out_text, mapping)

    # 6) حفظ إلى إكسل
    save_labeled_excel(labeled_df, OUT_XLSX)

if __name__ == "__main__":
    try:
        import openpyxl  # noqa: F401
    except ImportError:
        print("📦 تثبيت openpyxl...")
        import sys
        !pip install openpyxl -q
        import openpyxl  # noqa: F401

    main()
