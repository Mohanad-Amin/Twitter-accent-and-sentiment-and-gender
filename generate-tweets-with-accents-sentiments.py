import os
import re
import time
import json
import math
import random
import requests
import pandas as pd
from collections import deque
from google.colab import userdata
from google.colab import drive

"""
مولِّد تغريدات للهجات الشامية (سورية/لبنانية/أردنية/فلسطينية)
باستخدام GPT-4o عبر **OpenAI Batch API فقط**
(النتيجة: ملف Excel بثلاثة أعمدة فقط: التغريدة | المشاعر | اللهجة)
"""

# ==============================
# Google Drive
# ==============================
drive.mount('/content/drive')

# ==============================
# OpenAI API (Batch)
# ==============================
OPENAI_FILES_URL = "https://api.openai.com/v1/files"
OPENAI_BATCHES_URL = "https://api.openai.com/v1/batches"
OPENAI_CHAT_ENDPOINT = "/v1/chat/completions"
MODEL = "gpt-4o"
BATCH_COMPLETION_WINDOW = "24h"

# ==============================
# API KEY
# ==============================
API_KEY = ""
try:
    API_KEY = userdata.get('Mohanad') or userdata.get('OPENAI_API_KEY') or ""
except Exception:
    pass
if not API_KEY:
    API_KEY = os.environ.get('Mohanad', '') or os.environ.get('OPENAI_API_KEY', '')
if not API_KEY:
    print("❌ لم يتم العثور على مفتاح OpenAI (Mohanad / OPENAI_API_KEY)")
    raise SystemExit(1)

# ==============================
# بنوك التنويع للشام
# ==============================
DIALECTS = ["سورية", "لبنانية", "أردنية", "فلسطينية"]

DIALECT_HINTS = {
    "سورية":   "ممكن: شو/لسّا/هلّق/كتير/تمام/يعني/مو/بدّي/مشان — بلا مبالغة.",
    "لبنانية": "ممكن: شو/مزبوط/عنجد/هيك/قدّي/كتير/هلق/ما بعرف/خلص — بلا مبالغة.",
    "أردنية":  "ممكن: شو/لسّه/هسه/قدّيش/ليش/مزبوط/زلمة/قعدة/تمام — بلا مبالغة.",
    "فلسطينية":"ممكن: شو/لسّا/هلّق/زلمة/عنجد/منيح/قدّيش/طبعًا/خلص — بلا مبالغة."
}

TOPICS = [
    "العجقة والمواصلات", "السرفيس/الميكرو والتبديل", "القهوة على البلكون/التيراس", "المشاوير والطلعات",
    "المناقيش والفلافل والشاورما", "المقلوبة والمنسف والأكلات البيتية", "الجامعة والمحاضرات", "الشغل والدوام",
    "المولات والأسواق", "المباريات والكرة", "التسوق الإلكتروني", "السهرة مع الأصحاب",
    "الطقس (مطر/برد/شوب)", "المشاوير للجبل/الضيعة", "المسلسلات والسينيما", "الصبح بدري والنشاط"
]

STYLES   = ["ساخر", "لطيف", "متفائل", "مستعجل", "منزعج", "سؤال مباشر", "نوستالجيا", "تحفيزي", "حكائي قصير"]
PERSONAS = ["طالب/ـة جامعة", "موظف/ـة مكتب", "رياضي/ـة جيم", "مطور/ـة برمجيات", "أب/أم شاب", "مهووس/ـة قهوة", "شخص يحب السفر", "عاشق/ـة أكل"]
TIMES    = ["الصبح بدري", "قبل الدوام", "بعد الدوام", "بعد العشاء", "وقت الغروب", "وقت الشوب", "وقت المطر"]

SCENARIOS = [
    "واقف عالإشارة والزحمة خانقة", 
    "راكب سرفيس وعم يستنى يفضى كرسي", 
    "قعدة على البلكون مع فنجان قهوة", 
    "طلعة خفيفة عالجبل/مزرعة", 
    "واقف بالدور عفرن مناقيش",
    "داخل مول ويدوّر على خصم",
    "راجع من الدوام وتعبان",
    "يحضر لمباراة مع الشباب",
    "يستنى طلب دليفري",
    "يوثّق لحظة غروب حلوة",
    "مرتب سهرة بسيطة بالبيت"
]

recent_topics = deque(maxlen=24)

# ==============================
# Sentiment Utilities
# ==============================
SENTIMENT_LABELS = ["إيجابي", "سلبي", "محايد"]

def normalize_sentiment_label(v: str) -> str:
    if not v:
        return "غير محدد"
    v = str(v).strip().lower()
    m = {
        'positive': 'إيجابي', 'pos': 'إيجابي', 'ايجابي': 'إيجابي', 'إيجابي': 'إيجابي',
        'negative': 'سلبي', 'neg': 'سلبي', 'سلبي': 'سلبي',
        'neutral': 'محايد', 'neu': 'محايد', 'محايد': 'محايد'
    }
    return m.get(v, 'محايد')

def choose_non_repeat(options, memory: deque):
    pool = [o for o in options if o not in memory]
    if not pool:
        memory.clear()
        pool = options[:]
    pick = random.choice(pool)
    memory.append(pick)
    return pick

def build_context():
    topic   = choose_non_repeat(TOPICS, recent_topics)
    style   = random.choice(STYLES)
    persona = random.choice(PERSONAS)
    time_s  = random.choice(TIMES)
    scen    = random.choice(SCENARIOS)
    dialect = random.choice(DIALECTS)

    flags = {
        'hashtag': random.random() < 0.30,
        'emoji':   random.random() < 0.25,
        'english': random.random() < 0.20,
        'franco':  random.random() < 0.10,
    }

    return {
        'dialect': dialect,
        'topic': topic,
        'style': style,
        'persona': persona,
        'time': time_s,
        'city': "",
        'scenario': scen,
        'flags': flags,
        'dialect_hint': DIALECT_HINTS.get(dialect, "")
    }

def build_user_message(ctx: dict, target_sent: str) -> str:
    instr = (
        "اكتب تغريدة واحدة باللهجة الشامية ({dialect}) فقط.\n"
        "المشاعر المطلوبة: {sentiment}. يجب أن تُعبّر التغريدة بوضوح عن هذا الشعور.\n"
        "أعِد JSON صالح فقط بهذا الشكل: {{\"text\": \"...\", \"sentiment\": \"{sentiment}\"}}.\n"
        "القواعد: طول 60–200 حرف تقريبًا، لهجة شامية طبيعية وواضحة، بدون روابط أو أسماء/أرقام/منشنات حقيقية، "
        "علامات ترقيم طبيعية، وممنوع ذكر أي أسماء مدن/مناطق/دول نهائيًا.\n"
        "تلميحات أسلوبية اختيارية: {dialect_hint}\n"
        "طبّق الأعلام التالية حرفيًا: الهاشتاج: {hashflag} — الإيموجي: {emoflag} — إنجليزي: {engflag} — فرانكو: {fflag}.\n"
        "السياق: موضوع={topic}، أسلوب={style}، شخصية متكلمة={persona}، توقيت={time}، (بدون ذكر مدن/مناطق/دول)، سيناريو={scenario}.\n"
        "أعِد JSON فقط بدون أي شرح إضافي، وبالحقول: text, sentiment لا غير."
    ).format(
        dialect=ctx['dialect'], sentiment=target_sent, dialect_hint=ctx.get('dialect_hint', ''),
        topic=ctx['topic'], style=ctx['style'], persona=ctx['persona'], time=ctx['time'],
        scenario=ctx['scenario'],
        hashflag=("أضِف هاشتاج واحد مناسب" if ctx['flags']['hashtag'] else "ممنوع الهاشتاج"),
        emoflag=("مسموح بإيموجي واحد بحد أقصى اثنين" if ctx['flags']['emoji'] else "ممنوع الإيموجي"),
        engflag=("أدخل كلمة إنجليزية واحدة مناسبة" if ctx['flags']['english'] else "ممنوع الكلمات الإنجليزية"),
        fflag=("مسموح بكلمة فرانكو واحدة على الأكثر" if ctx['flags']['franco'] else "ممنوع الفرانكو")
    )
    return instr

# ==============================
# Allocation of target sentiments
# ==============================
def normalize_weights(weights: dict) -> dict:
    w = {lbl: float(max(0.0, weights.get(lbl, 0.0))) for lbl in SENTIMENT_LABELS}
    s = sum(w.values())
    if s <= 0:
        n = len(SENTIMENT_LABELS)
        return {lbl: 1.0 / n for lbl in SENTIMENT_LABELS}
    return {k: v / s for k, v in w.items()}

def build_sentiment_plan(num_items: int, weights: dict, mode: str = 'deterministic') -> list:
    weights = normalize_weights(weights)
    labels = SENTIMENT_LABELS
    if mode not in ('deterministic', 'probabilistic'):
        mode = 'deterministic'
    if mode == 'probabilistic':
        probs = [weights[lbl] for lbl in labels]
        return random.choices(labels, weights=probs, k=num_items)

    raw_counts = {lbl: num_items * weights[lbl] for lbl in labels}
    floor_counts = {lbl: int(math.floor(raw_counts[lbl])) for lbl in labels}
    assigned = sum(floor_counts.values())
    remainder = num_items - assigned
    fracs = sorted(((lbl, raw_counts[lbl] - floor_counts[lbl]) for lbl in labels), key=lambda x: x[1], reverse=True)
    for i in range(remainder):
        floor_counts[fracs[i % len(fracs)][0]] += 1
    plan = []
    for lbl in labels:
        plan.extend([lbl] * floor_counts[lbl])
    random.shuffle(plan)
    return plan[:num_items]

# ==============================
# مانع أسماء المدن/الدول
# ==============================
CITY_NAME_BLACKLIST = [
    # سوريا
    "دمشق","الشام","حلب","حمص","حماة","اللاذقية","طرطوس","درعا","السويداء","إدلب","الرقة","دير الزور","الحسكة","القامشلي",
    # لبنان
    "بيروت","طرابلس","صيدا","صور","زحلة","بعلبك","جبيل","جونية","النبطية",
    # الأردن
    "عمّان","عمان","الزرقاء","إربد","العقبة","مادبا","السلط","جرش","عجلون","الكرك","معان","الطفيلة",
    # فلسطين
    "القدس","رام الله","غزة","نابلس","الخليل","بيت لحم","طولكرم","جنين","رفح","خان يونس","قلقيلية",
    # دول
    "سوريا","لبنان","الأردن","الاردن","فلسطين"
]
DELIMS = r"[\s\.,!?؛،:\-–—…\(\)\[\]\"'«»]"
GEO_PATTERN = re.compile(
    r"(^|" + DELIMS + r")(" + "|".join(map(re.escape, CITY_NAME_BLACKLIST)) + r")(?=($|" + DELIMS + r"))",
    flags=re.IGNORECASE
)

def _clean_spaces_punct(s: str) -> str:
    s = re.sub(r"\s{2,}", " ", s)
    s = re.sub(r"\s*([،,\.!?…؛:])\s*", r" \1 ", s)
    return s.strip()

def remove_geo_names(text: str) -> str:
    if text is None:
        return text
    s = str(text)
    s2 = GEO_PATTERN.sub(lambda m: m.group(1), s)
    if s2 != s:
        s2 = _clean_spaces_punct(s2)
    return s2

def normalize_text(s: str) -> str:
    s = s.strip()
    if s.startswith('"') and s.endswith('"'):
        s = s[1:-1]
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"https?://\S+", "", s)
    s = re.sub(r"@[A-Za-z0-9_]+", "", s)
    s = remove_geo_names(s)
    return s.strip()

# ==============================
# JSONL / Mapping
# ==============================
def prepare_jsonl_for_batch(num_items: int, jsonl_path: str, mapping_path: str,
                            sentiment_weights: dict, allocation_mode: str = 'deterministic'):
    plan = build_sentiment_plan(num_items, sentiment_weights, allocation_mode)

    from collections import Counter
    cnt = Counter(plan)
    total = sum(cnt.values()) or 1
    print("📐 خطة المشاعر قبل الإرسال:")
    for lbl in SENTIMENT_LABELS:
        print(f"  - {lbl}: {cnt.get(lbl, 0)} ({(cnt.get(lbl, 0)/total*100):.2f}%)")

    with open(jsonl_path, 'w', encoding='utf-8') as jf, open(mapping_path, 'w', encoding='utf-8') as mf:
        ts = int(time.time() * 1000)
        for i in range(num_items):
            ctx = build_context()
            target_sent = plan[i]
            custom_id = f"tw-levant-sent-{ts}-{i+1}"
            user_msg = build_user_message(ctx, target_sent)
            body = {
                "custom_id": custom_id,
                "method": "POST",
                "url": OPENAI_CHAT_ENDPOINT,
                "body": {
                    "model": MODEL,
                    "messages": [
                        {"role": "system", "content": "أنت مساعد يكتب تغريدات باللهجات الشامية بدقة وبدون شرح."},
                        {"role": "user",   "content": user_msg}
                    ],
                    "response_format": {"type": "json_object"},
                    "temperature": 0.95,
                    "top_p": 0.95,
                    "max_tokens": 220
                }
            }
            jf.write(json.dumps(body, ensure_ascii=False) + "\n")
            mf.write(json.dumps({"custom_id": custom_id, "ctx": ctx, "target_sentiment": target_sent}, ensure_ascii=False) + "\n")
    print(f"🗂️ JSONL: {jsonl_path}")
    print(f"🗺️ Mapping: {mapping_path}")

# ==============================
# Batch API helpers
# ==============================
def upload_file_for_batch(jsonl_path: str) -> str:
    headers = {"Authorization": f"Bearer {API_KEY}"}
    files = {"file": (os.path.basename(jsonl_path), open(jsonl_path, 'rb'))}
    data = {"purpose": "batch"}
    r = requests.post(OPENAI_FILES_URL, headers=headers, files=files, data=data, timeout=120)
    if r.status_code != 200:
        raise RuntimeError(f"File upload failed: {r.status_code} {r.text}")
    file_id = r.json()["id"]
    print(f"📎 تم الرفع: file_id={file_id}")
    return file_id

def create_batch_job(file_id: str, completion_window: str = BATCH_COMPLETION_WINDOW) -> str:
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    payload = {
        "input_file_id": file_id,
        "endpoint": OPENAI_CHAT_ENDPOINT,
        "completion_window": completion_window
    }
    r = requests.post(OPENAI_BATCHES_URL, headers=headers, json=payload, timeout=60)
    if r.status_code != 200:
        raise RuntimeError(f"Create batch failed: {r.status_code} {r.text}")
    batch_id = r.json()["id"]
    print(f"📦 تم إنشاء Batch: batch_id={batch_id}")
    return batch_id

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
    r = requests.get(f"{OPENAI_FILES_URL}/{file_id}/content", headers=headers, timeout=120)
    if r.status_code != 200:
        raise RuntimeError(f"Download file failed: {r.status_code} {r.text}")
    return r.text

# ==============================
# Mapping loader
# ==============================
def load_mapping_jsonl(mapping_path: str) -> dict:
    mapping = {}
    with open(mapping_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            mapping[row.get('custom_id')] = row
    return mapping

# ==============================
# Parse batch output  -> (نخزّن فقط ما نحتاجه)
# ==============================
def parse_batch_output(jsonl_text: str, mapping: dict) -> list:
    tweets = []
    for line in jsonl_text.splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)

        if obj.get("error"):
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
            content_obj = json.loads(content) if isinstance(content, str) else content
        except json.JSONDecodeError:
            continue

        text = normalize_text(str(content_obj.get("text", "")).strip())
        sentiment_raw = content_obj.get("sentiment", "")
        sentiment = normalize_sentiment_label(sentiment_raw)
        if not text:
            continue

        custom_id = obj.get("custom_id")
        meta = mapping.get(custom_id, {}) if custom_id else {}
        ctx = meta.get('ctx', {}) if meta else {}

        tweets.append({
            'التغريدة': text,
            'المشاعر': sentiment if sentiment in SENTIMENT_LABELS else "محايد",
            'اللهجة': ctx.get('dialect', '')
        })

    # ترقيم داخلي لو احتجته لاحقًا (غير مكتوب للملف)
    for i, row in enumerate(tweets, start=1):
        row['_idx'] = i

    return tweets

# ==============================
# Save to Excel (3 أعمدة فقط)
# ==============================
def save_to_excel(tweets: list, filename: str = "levant_tweets_MINIMAL_3cols.xlsx"):
    if not tweets:
        print("❌ لا توجد تغريدات للحفظ!")
        return
    df = pd.DataFrame(tweets)
    # ثلاثة أعمدة فقط وبالترتيب المطلوب
    df = df[['التغريدة', 'المشاعر', 'اللهجة']].copy()

    file_path = f"/content/drive/MyDrive/{filename}"
    try:
        with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='التغريدات', index=False)
        print(f"✅ تم حفظ {len(df)} تغريدة في: {file_path}")
        print("📄 الأعمدة: التغريدة | المشاعر | اللهجة")
    except Exception as e:
        print(f"❌ خطأ في الحفظ: {e}")

# ==============================
# Main
# ==============================
def main():
    print("🚀 مولد تغريدات شامية — GPT-4o — Batch API — (Excel بثلاثة أعمدة فقط)")
    print("=" * 100)

    # ---- إعدادات ----
    NUM_TWEETS = 100
    SENTIMENT_WEIGHTS = {'إيجابي': 0.3, 'سلبي': 0.4, 'محايد': 0.3}
    ALLOCATION_MODE = 'deterministic'

    JSONL_PATH = "/content/drive/MyDrive/levant_MIN_requests.jsonl"
    MAPPING_PATH = "/content/drive/MyDrive/levant_MIN_mapping.jsonl"
    RAW_OUTPUT_JSONL_PATH = "/content/drive/MyDrive/levant_MIN_output.jsonl"

    # 1) تجهيز الطلبات
    prepare_jsonl_for_batch(
        num_items=NUM_TWEETS,
        jsonl_path=JSONL_PATH,
        mapping_path=MAPPING_PATH,
        sentiment_weights=SENTIMENT_WEIGHTS,
        allocation_mode=ALLOCATION_MODE,
    )

    # 2) رفع الملف وإنشاء الباتش
    file_id = upload_file_for_batch(JSONL_PATH)
    batch_id = create_batch_job(file_id, completion_window=BATCH_COMPLETION_WINDOW)

    # 3) الانتظار
    status_obj = poll_batch_until_done(batch_id, poll_interval=20)
    final_status = status_obj.get("status")
    print(f"✅ الحالة النهائية: {final_status}")
    if final_status != "completed":
        print("❌ لم يكتمل الباتش بنجاح. التفاصيل:", status_obj)
        return

    # 4) تنزيل الإخراج
    output_file_id = status_obj.get("output_file_id")
    if not output_file_id:
        print("❌ لم يتم العثور على output_file_id.")
        return

    output_jsonl = download_file_content(output_file_id)
    with open(RAW_OUTPUT_JSONL_PATH, 'w', encoding='utf-8') as f:
        f.write(output_jsonl)
    print(f"📥 حفظ الخام في: {RAW_OUTPUT_JSONL_PATH}")

    # 5) تحليل النتائج -> ثلاثة أعمدة فقط
    mapping = load_mapping_jsonl(MAPPING_PATH)
    tweets = parse_batch_output(output_jsonl, mapping)
    if not tweets:
        print("❌ لم تُستخرج تغريدات من نتائج الباتش.")
        return

    # 6) حفظ Excel (3 أعمدة فقط)
    save_to_excel(tweets, filename="levant_tweets_MINIMAL_3cols.xlsx")

# ==============================
# Entrypoint
# ==============================
if __name__ == "__main__":
    try:
        import openpyxl  # noqa: F401
    except ImportError:
        print("📦 تثبيت openpyxl...")
        import sys
        # يعمل داخل خلية كولاب
        !pip install openpyxl -q
        import openpyxl  # noqa: F401

    main()
