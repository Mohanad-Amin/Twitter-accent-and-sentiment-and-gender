# === Cell 1: الإعدادات والاستيراد ===
!pip install -q datasets evaluate

from google.colab import drive
drive.mount('/content/drive')

# قائمة مسارات ملفات البيانات
DATA_PATHS = [
    "/content/drive/MyDrive/Tweet_Project/Data/Arabic Sentiment Analysis 1_labelled.xlsx",
    "/content/drive/MyDrive/Tweet_Project/Data/full_tweets_clean2.xlsx",
    #"/content/drive/MyDrive/Tweet_Project/Data/Arabic Sentiment Analysis 3_labelled.xlsx",
]

OUT_DIR = "/content/drive/MyDrive/Tweet_Project/Models/sentiment_marbert_v308_improved"
MODEL_NAME = "UBC-NLP/MARBERTv2"

# تحسين MAX_LEN بناءً على التحليل (99% من النصوص أقل من 99 رمز)
MAX_LEN = 100  # بدلاً من 128 - سيوفر ذاكرة ووقت
SEED = 42

import os, re, json, random
import numpy as np
import torch
os.makedirs(OUT_DIR, exist_ok=True)

# ضبط البذور للحصول على نتائج قابلة للتكرار
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(SEED)

import pandas as pd
import json
from tqdm import tqdm

def read_single_file(file_path):
    """قراءة ملف واحد واستخراج عمودي text و sentiment فقط"""
    print(f"📁 قراءة الملف: {file_path}")

    if not os.path.exists(file_path):
        print(f"⚠️ تحذير: الملف غير موجود: {file_path}")
        return pd.DataFrame()

    try:
        df = pd.read_excel(file_path)
        print(f"   شكل البيانات: {df.shape}")
        print(f"   الأعمدة: {df.columns.tolist()}")

        # التحقق من وجود العمودين المطلوبين
        if "text" not in df.columns:
            print(f"❌ خطأ: عمود 'text' غير موجود في {file_path}")
            return pd.DataFrame()

        if "sentiment" not in df.columns:
            print(f"❌ خطأ: عمود 'sentiment' غير موجود في {file_path}")
            return pd.DataFrame()

        # أخذ العمودين المطلوبين فقط
        df = df[["text", "sentiment"]].copy()

        # إضافة معلومات المصدر
        df["source_file"] = os.path.basename(file_path)

        # تنظيف أولي للبيانات
        initial_count = len(df)
        df = df.dropna(subset=["text", "sentiment"])
        df = df[df["text"].astype(str).str.strip() != ""]
        df = df[df["sentiment"].astype(str).str.strip() != ""]

        # إزالة النصوص القصيرة جداً (أقل من 10 أحرف)
        df = df[df["text"].str.len() >= 10]

        final_count = len(df)

        print(f"   📊 البيانات الصالحة: {final_count:,}/{initial_count:,}")

        if len(df) > 0:
            sample_text = df["text"].iloc[0]
            sample_sentiment = df["sentiment"].iloc[0]
            print(f"   📝 عينة: {sample_text[:50]}... -> {sample_sentiment}")
            print(f"   ✅ تم تحميل البيانات بنجاح")
        else:
            print(f"   ⚠️ تحذير: لا توجد بيانات صالحة في الملف")

        return df

    except Exception as e:
        print(f"❌ خطأ في قراءة الملف {file_path}: {e}")
        return pd.DataFrame()

# قراءة جميع الملفات ودمجها
all_dataframes = []
total_original_rows = 0
file_stats = []

print("🔄 بدء قراءة الملفات...")
print("=" * 60)

for i, file_path in enumerate(DATA_PATHS, 1):
    print(f"\n[{i}/{len(DATA_PATHS)}] معالجة الملف:")
    df_single = read_single_file(file_path)

    if not df_single.empty:
        all_dataframes.append(df_single)
        total_original_rows += len(df_single)

        file_stat = {
            "file_path": file_path,
            "file_name": os.path.basename(file_path),
            "rows_count": len(df_single),
            "has_source_column": "source_file" in df_single.columns
        }

        file_stats.append(file_stat)

        print(f"   📊 عدد الأسطر: {len(df_single):,}")

# التحقق من وجود بيانات
if not all_dataframes:
    raise ValueError("❌ لم يتم العثور على أي بيانات صالحة في الملفات المحددة")

# دمج جميع البيانات
print(f"\n📋 دمج البيانات من {len(all_dataframes)} ملف(ات)...")
df = pd.concat(all_dataframes, ignore_index=True)

print(f"إجمالي الأسطر قبل التنظيف: {total_original_rows:,}")
print(f"إجمالي الأسطر بعد الدمج: {len(df):,}")

# === Cell 2: تنظيف وتحسين البيانات ===

# تنظيف محسّن للنصوص مع الحفاظ على الرموز التعبيرية
def enhanced_sentiment_clean(text):
    if not isinstance(text, str):
        return ""

    # حفظ الإيموجيز قبل التنظيف
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE
    )
    emojis = emoji_pattern.findall(text)

    # إزالة الروابط والمنشن
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"@[A-Za-z0-9_]+", " ", text)
    text = re.sub(r"#(\w+)", r" \1 ", text)  # الحفاظ على محتوى الهاشتاج

    # إزالة التطويل مع الحفاظ على التكرار المعبر (مثل !!! أو ...)
    text = re.sub(r"([أ-ي])\1{2,}", r"\1\1", text)  # تطويل الأحرف العربية
    text = re.sub(r"([!?.])\1{3,}", r"\1\1\1", text)  # الحفاظ على 3 تكرارات كحد أقصى

    # تنظيف المسافات
    text = re.sub(r"\s+", " ", text).strip()

    # إعادة إضافة الإيموجيز في النهاية
    if emojis:
        text = text + " " + " ".join(emojis)

    return text

df["text"] = df["text"].apply(enhanced_sentiment_clean)

# إزالة الأسطر الفارغة والمكررة
print("🧹 تنظيف البيانات المدمجة...")

before_cleaning = len(df)
df = df.dropna(subset=["text","sentiment"])
df = df[df["text"].str.len() >= 10]  # إزالة النصوص القصيرة جداً
df = df.drop_duplicates(subset=["text"], keep="first")
df = df.reset_index(drop=True)

print(f"   قبل التنظيف: {before_cleaning:,} سطر")
print(f"   بعد التنظيف: {len(df):,} سطر")
print(f"   إجمالي المحذوف: {before_cleaning - len(df):,} سطر")

# تحويل تسميات المشاعر إلى أرقام - محسّن
sentiment_mapping = {
    # العربية
    "positive": 2, "إيجابي": 2, "ايجابي": 2, "إيجابية": 2, "ايجابية": 2,
    "negative": 0, "سلبي": 0, "سالب": 0, "سلبية": 0, "سالبة": 0,
    "neutral": 1, "محايد": 1, "متوسط": 1, "محايدة": 1,
    # الإنجليزية
    "pos": 2, "neg": 0, "neu": 1,
    # الأرقام
    1: 2, 2: 2,     # إيجابي
    -1: 0, 0: 0,    # سلبي
}

def map_sentiment(sentiment):
    if isinstance(sentiment, str):
        sentiment = sentiment.strip().lower()

    if sentiment in sentiment_mapping:
        return sentiment_mapping[sentiment]

    # محاولة تحويل إلى رقم
    try:
        val = float(sentiment)
        if val > 0:
            return 2  # إيجابي
        elif val < 0:
            return 0  # سلبي
        else:
            return 1  # محايد
    except:
        pass

    # التحقق من الكلمات المفتاحية
    sentiment_str = str(sentiment).lower()
    if any(word in sentiment_str for word in ["positive", "pos", "إيجاب", "ايجاب", "جيد", "ممتاز", "رائع"]):
        return 2
    elif any(word in sentiment_str for word in ["negative", "neg", "سلب", "سيء", "فاسد", "سوء"]):
        return 0
    else:
        return 1  # محايد كقيمة افتراضية

df["labels"] = df["sentiment"].apply(map_sentiment)
df = df.dropna(subset=["labels"]).reset_index(drop=True)
df["labels"] = df["labels"].astype(int)

# التأكد من أن القيم ضمن النطاق المتوقع
valid_labels = set([0, 1, 2])
seen = set(df["labels"].unique().tolist())
assert seen.issubset(valid_labels), f"وجدت قيم ليبل غير متوقعة: {sorted(seen - valid_labels)}"

# خرائط فئات المشاعر
id2label = {
    0: "سلبي",      # Negative
    1: "محايد",     # Neutral
    2: "إيجابي"     # Positive
}
label2id = {v:k for k,v in id2label.items()}

print("\n📊 توزيع المشاعر:")
sentiment_counts = df["labels"].value_counts().sort_index()
for idx, count in sentiment_counts.items():
    print(f"{id2label[idx]}: {count:,} ({count/len(df)*100:.1f}%)")

# === Cell 3: معالجة عدم التوازن في البيانات ===

# موازنة البيانات باستخدام undersampling للفئة الأكثر
from sklearn.utils import resample

print("\n⚖️ موازنة البيانات...")

# الحصول على أقل عدد في الفئات
min_class_count = sentiment_counts.min()
target_count = int(min_class_count * 1.1)  # نأخذ 110% من أقل فئة

balanced_dfs = []
for label in [0, 1, 2]:
    df_class = df[df['labels'] == label]
    if len(df_class) > target_count:
        # Undersample
        df_class_sampled = resample(df_class,
                                   n_samples=target_count,
                                   random_state=SEED)
    else:
        # Oversample إذا كانت الفئة أقل من الهدف
        df_class_sampled = resample(df_class,
                                   n_samples=target_count,
                                   replace=True,
                                   random_state=SEED)
    balanced_dfs.append(df_class_sampled)

df_balanced = pd.concat(balanced_dfs)
df_balanced = df_balanced.sample(frac=1, random_state=SEED).reset_index(drop=True)

print(f"البيانات بعد الموازنة: {len(df_balanced):,} عينة")
print("التوزيع الجديد:")
for label, count in df_balanced['labels'].value_counts().sort_index().items():
    print(f"  {id2label[label]}: {count:,} ({count/len(df_balanced)*100:.1f}%)")

# استخدام البيانات المتوازنة
df = df_balanced

# === Cell 4: التقسيم والترميز ===
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict

# تقسيم stratified 70/15/15 (أفضل للبيانات المتوازنة)
train_columns = ["text", "labels"]

train_df, temp_df = train_test_split(
    df[train_columns], test_size=0.3, stratify=df["labels"], random_state=SEED
)
val_df, test_df = train_test_split(
    temp_df, test_size=0.5, stratify=temp_df["labels"], random_state=SEED
)

print(f"\n📊 تقسيم البيانات:")
print(f"Train: {len(train_df):,} ({len(train_df)/len(df)*100:.1f}%)")
print(f"Val: {len(val_df):,} ({len(val_df)/len(df)*100:.1f}%)")
print(f"Test: {len(test_df):,} ({len(test_df)/len(df)*100:.1f}%)")

ds = DatasetDict({
    "train": Dataset.from_pandas(train_df.reset_index(drop=True)),
    "validation": Dataset.from_pandas(val_df.reset_index(drop=True)),
    "test": Dataset.from_pandas(test_df.reset_index(drop=True)),
})

# === Cell 5: إعداد النموذج والتدريب المحسّن ===
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    DataCollatorWithPadding
)
import evaluate

# تحميل التوكينايزر والنموذج
tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
num_labels = len(id2label)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id,
    hidden_dropout_prob=0.2,  # إضافة dropout للتعميم
    attention_probs_dropout_prob=0.2
)

# دالة الترميز المحسّنة
def tokenize(batch):
    return tok(
        batch["text"],
        truncation=True,
        max_length=MAX_LEN,
        padding=False  # سنستخدم dynamic padding
    )

# ترميز البيانات
print("🔄 ترميز البيانات...")
ds_enc = ds.map(tokenize, batched=True, remove_columns=["text"])

# Data collator للـ dynamic padding
data_collator = DataCollatorWithPadding(tokenizer=tok, padding=True)

# المقاييس المحسّنة
f1_metric = evaluate.load("f1")
acc_metric = evaluate.load("accuracy")
precision_metric = evaluate.load("precision")
recall_metric = evaluate.load("recall")

def compute_metrics(eval_pred):
    logits, labels_np = eval_pred
    preds = logits.argmax(axis=-1)

    results = {}
    results["accuracy"] = acc_metric.compute(predictions=preds, references=labels_np)["accuracy"]
    results["macro_f1"] = f1_metric.compute(predictions=preds, references=labels_np, average="macro")["f1"]
    results["weighted_f1"] = f1_metric.compute(predictions=preds, references=labels_np, average="weighted")["f1"]

    # F1 لكل فئة
    for i, label_name in id2label.items():
        label_f1 = f1_metric.compute(
            predictions=preds,
            references=labels_np,
            labels=[i],
            average="micro"
        )["f1"]
        results[f"f1_{label_name}"] = label_f1

    results["macro_precision"] = precision_metric.compute(predictions=preds, references=labels_np, average="macro")["precision"]
    results["macro_recall"] = recall_metric.compute(predictions=preds, references=labels_np, average="macro")["recall"]

    return results

# إعدادات التدريب المحسّنة
training_args = TrainingArguments(
    output_dir=OUT_DIR,
    eval_strategy="steps",
    eval_steps=200,  # تقييم أكثر تكراراً
    save_strategy="steps",
    save_steps=200,
    save_total_limit=3,
    learning_rate=2e-5,  # معدل تعلم أقل للاستقرار
    per_device_train_batch_size=32,  # batch size متوسط
    per_device_eval_batch_size=64,
    num_train_epochs=5,  # epochs معتدل مع early stopping
    weight_decay=0.01,
    warmup_ratio=0.1,
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",
    greater_is_better=True,
    fp16=True,  # استخدام FP16 للسرعة
    gradient_checkpointing=True,
    gradient_accumulation_steps=2,  # لمحاكاة batch size أكبر
    dataloader_drop_last=False,
    eval_accumulation_steps=1,
    push_to_hub=False,
    report_to="none",  # تعطيل التقارير لتوفير الذاكرة
    optim="adamw_torch",  # محسّن أفضل
    lr_scheduler_type="cosine",  # جدولة معدل التعلم
    seed=SEED,
    data_seed=SEED,
    label_smoothing_factor=0.1,  # تنعيم التسميات للتعميم
)

# الـ Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=ds_enc["train"],
    eval_dataset=ds_enc["validation"],
    tokenizer=tok,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[
        EarlyStoppingCallback(
            early_stopping_patience=3,
            early_stopping_threshold=0.001
        )
    ]
)

# بدء التدريب
print("🚀 بدء تدريب نموذج تحليل المشاعر المحسّن...")
train_result = trainer.train()

# حفظ النموذج النهائي
print("💾 حفظ النموذج...")
trainer.save_model(OUT_DIR)
tok.save_pretrained(OUT_DIR)

# حفظ معلومات التدريب
training_info = {
    "id2label": id2label,
    "label2id": label2id,
    "num_labels": num_labels,
    "task": "sentiment_analysis",
    "language": "arabic",
    "model_name": MODEL_NAME,
    "max_length": MAX_LEN,
    "training_samples": len(train_df),
    "validation_samples": len(val_df),
    "test_samples": len(test_df),
    "train_runtime": train_result.metrics["train_runtime"],
    "train_samples_per_second": train_result.metrics["train_samples_per_second"],
    "train_loss": train_result.metrics["train_loss"],
    "data_info": {
        "original_samples": total_original_rows,
        "cleaned_samples": before_cleaning,
        "balanced_samples": len(df),
        "avg_text_length": df["text"].str.len().mean(),
        "sentiment_distribution": df["labels"].value_counts().to_dict()
    }
}

with open(os.path.join(OUT_DIR, "training_info.json"), "w", encoding="utf-8") as f:
    json.dump(training_info, f, ensure_ascii=False, indent=2)

print("✅ تم حفظ نموذج تحليل المشاعر المحسّن في:", OUT_DIR)

# === Cell 6: التقييم الشامل ===
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# إعداد الخط العربي
import matplotlib.font_manager as fm
# محاولة إيجاد خط عربي
arabic_fonts = [f for f in fm.findSystemFonts() if 'arabic' in f.lower() or 'arial' in f.lower()]
if arabic_fonts:
    plt.rcParams['font.family'] = fm.FontProperties(fname=arabic_fonts[0]).get_name()

# تقييم على مجموعة الاختبار
print("\n🔍 تقييم النموذج على بيانات الاختبار...")
pred = trainer.predict(ds_enc["test"])
preds = np.argmax(pred.predictions, axis=-1)
true = pred.label_ids

# أسماء فئات المشاعر
target_names = [id2label[i] for i in range(num_labels)]

# تقرير التصنيف المفصل
report = classification_report(true, preds, target_names=target_names, digits=4)
print("\n📊 تقرير تحليل المشاعر:")
print("=" * 50)
print(report)

# تقرير مفصل بصيغة dictionary
report_dict = classification_report(true, preds, target_names=target_names, output_dict=True)

# حساب المقاييس الإجمالية
test_metrics = {
    "test_accuracy": float(acc_metric.compute(predictions=preds, references=true)["accuracy"]),
    "test_macro_f1": float(f1_metric.compute(predictions=preds, references=true, average="macro")["f1"]),
    "test_weighted_f1": float(f1_metric.compute(predictions=preds, references=true, average="weighted")["f1"]),
    "test_macro_precision": float(precision_metric.compute(predictions=preds, references=true, average="macro")["precision"]),
    "test_macro_recall": float(recall_metric.compute(predictions=preds, references=true, average="macro")["recall"]),
    "n_test_samples": int(len(true)),
    "per_class_metrics": {
        label_name: {
            "precision": report_dict[label_name]["precision"],
            "recall": report_dict[label_name]["recall"],
            "f1-score": report_dict[label_name]["f1-score"],
            "support": report_dict[label_name]["support"]
        }
        for label_name in target_names
    }
}

print("\n📈 المقاييس الإجمالية:")
print("=" * 30)
for metric, value in test_metrics.items():
    if metric not in ["n_test_samples", "per_class_metrics"]:
        print(f"{metric}: {value:.4f}")
    elif metric == "n_test_samples":
        print(f"{metric}: {value}")

# حفظ التقرير والمقاييس
with open(os.path.join(OUT_DIR, "evaluation_report.txt"), "w", encoding="utf-8") as f:
    f.write("تقرير تقييم نموذج تحليل المشاعر المحسّن\n")
    f.write("=" * 50 + "\n\n")
    f.write(report + "\n\n")
    f.write("المقاييس الإجمالية:\n")
    f.write(json.dumps(test_metrics, ensure_ascii=False, indent=2))

# رسم مصفوفة الالتباس المحسّنة
cm = confusion_matrix(true, preds, labels=list(range(num_labels)))

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=target_names, yticklabels=target_names,
            cbar_kws={'label': 'عدد العينات'})
plt.title('مصفوفة الالتباس - تحليل المشاعر', fontsize=16, pad=20)
plt.xlabel('التوقع', fontsize=14)
plt.ylabel('الحقيقة', fontsize=14)

# إضافة نسب مئوية
for i in range(len(cm)):
    for j in range(len(cm)):
        percentage = cm[i, j] / cm[i].sum() * 100
        plt.text(j + 0.5, i + 0.7, f'{percentage:.1f}%',
                ha='center', va='center', fontsize=9, color='gray')

plt.tight_layout()
cm_path = os.path.join(OUT_DIR, "confusion_matrix.png")
plt.savefig(cm_path, dpi=300, bbox_inches='tight')
plt.show()

# رسم مقاييس الأداء لكل فئة
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# الرسم الأول: F1-Score لكل فئة
f1_scores = [test_metrics["per_class_metrics"][label]["f1-score"] for label in target_names]
bars1 = ax1.bar(target_names, f1_scores, color=['red', 'gray', 'green'], alpha=0.7)
ax1.set_title('F1-Score لكل فئة', fontsize=14)
ax1.set_ylabel('F1-Score', fontsize=12)
ax1.set_ylim(0, 1)
ax1.grid(axis='y', alpha=0.3)

# إضافة القيم على الأعمدة
for bar, score in zip(bars1, f1_scores):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{score:.3f}', ha='center', va='bottom')

# الرسم الثاني: Precision و Recall
x = np.arange(len(target_names))
width = 0.35

precision_scores = [test_metrics["per_class_metrics"][label]["precision"] for label in target_names]
recall_scores = [test_metrics["per_class_metrics"][label]["recall"] for label in target_names]

bars2 = ax2.bar(x - width/2, precision_scores, width, label='Precision', alpha=0.7)
bars3 = ax2.bar(x + width/2, recall_scores, width, label='Recall', alpha=0.7)

ax2.set_title('Precision و Recall لكل فئة', fontsize=14)
ax2.set_ylabel('القيمة', fontsize=12)
ax2.set_xticks(x)
ax2.set_xticklabels(target_names)
ax2.legend()
ax2.set_ylim(0, 1)
ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
metrics_path = os.path.join(OUT_DIR, "performance_metrics.png")
plt.savefig(metrics_path, dpi=300, bbox_inches='tight')
plt.show()

# === Cell 7: اختبار النموذج وحفظ أمثلة ===
def test_sentiment_model(texts, model_path=OUT_DIR):
    """اختبار النموذج على نصوص جديدة"""
    from transformers import pipeline

    # تحميل النموذج كـ pipeline
    classifier = pipeline(
        "text-classification",
        model=model_path,
        tokenizer=model_path,
        device=0 if torch.cuda.is_available() else -1,
        max_length=MAX_LEN,
        truncation=True
    )

    results = []
    for text in texts:
        # الحصول على جميع الاحتماليات
        result = classifier(text, top_k=None)

        # ترتيب النتائج حسب الاحتمالية
        result_sorted = sorted(result, key=lambda x: x['score'], reverse=True)

        results.append({
            "text": text,
            "predicted_sentiment": result_sorted[0]["label"],
            "confidence": result_sorted[0]["score"],
            "all_scores": {r["label"]: r["score"] for r in result_sorted}
        })

    return results

# نصوص تجريبية متنوعة للاختبار
test_texts = [
    "أحب هذا الفيلم كثيراً، إنه رائع جداً! 😍",
    "هذا المطعم سيء للغاية، لن أعود إليه مرة أخرى 😠",
    "الطقس اليوم عادي، لا هو حار ولا بارد",
    "شكراً لك على المساعدة، أقدر جهودك كثيراً ❤️",
    "لا أعرف ماذا أقول عن هذا الموضوع",
    "الخدمة كانت بطيئة لكن الطعام لذيذ",
    "أسوأ تجربة في حياتي! 😤",
    "منتج عادي، لا يستحق السعر المدفوع",
    "ممتاز! تجاوز كل توقعاتي 🌟",
    "المنتج وصل متأخراً ومعطوباً، خدمة العملاء لم تساعد"
]

print("\n🧪 اختبار النموذج على نصوص جديدة:")
print("=" * 80)
test_results = test_sentiment_model(test_texts)

# عرض النتائج بشكل مفصل
for i, result in enumerate(test_results, 1):
    print(f"\n{i}. النص: {result['text']}")
    print(f"   🎭 المشاعر المتوقعة: {result['predicted_sentiment']}")
    print(f"   📊 مستوى الثقة: {result['confidence']:.3f}")
    print(f"   📈 جميع الاحتماليات:")
    for sentiment, score in result['all_scores'].items():
        bar = "█" * int(score * 20)
        print(f"      {sentiment}: {bar} {score:.3f}")

# حفظ أمثلة الاختبار
test_examples = {
    "test_date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
    "model_path": OUT_DIR,
    "examples": test_results
}

with open(os.path.join(OUT_DIR, "test_examples.json"), "w", encoding="utf-8") as f:
    json.dump(test_examples, f, ensure_ascii=False, indent=2)

# === Cell 8: تحليل الأخطاء ===
print("\n🔍 تحليل الأخطاء...")

# الحصول على الأخطاء من مجموعة الاختبار
errors = []
for i, (pred_label, true_label) in enumerate(zip(preds, true)):
    if pred_label != true_label:
        text = test_df.iloc[i]["text"]
        errors.append({
            "text": text,
            "true_label": id2label[true_label],
            "pred_label": id2label[pred_label],
            "text_length": len(text)
        })

print(f"\nعدد الأخطاء: {len(errors)} من أصل {len(test_df)} ({len(errors)/len(test_df)*100:.1f}%)")

# تحليل أنواع الأخطاء
error_types = {}
for error in errors:
    error_type = f"{error['true_label']} → {error['pred_label']}"
    if error_type not in error_types:
        error_types[error_type] = []
    error_types[error_type].append(error)

print("\n📊 توزيع أنواع الأخطاء:")
for error_type, error_list in sorted(error_types.items(), key=lambda x: len(x[1]), reverse=True):
    print(f"   {error_type}: {len(error_list)} خطأ ({len(error_list)/len(errors)*100:.1f}%)")

# عرض أمثلة من الأخطاء الأكثر شيوعاً
print("\n📝 أمثلة من الأخطاء:")
for error_type, error_list in sorted(error_types.items(), key=lambda x: len(x[1]), reverse=True)[:3]:
    print(f"\n{error_type}:")
    for error in error_list[:2]:  # عرض مثالين فقط
        print(f"   - {error['text'][:80]}...")

# حفظ تحليل الأخطاء
error_analysis = {
    "total_errors": len(errors),
    "error_rate": len(errors) / len(test_df),
    "error_types": {k: len(v) for k, v in error_types.items()},
    "sample_errors": errors[:20]  # حفظ 20 مثال فقط
}

with open(os.path.join(OUT_DIR, "error_analysis.json"), "w", encoding="utf-8") as f:
    json.dump(error_analysis, f, ensure_ascii=False, indent=2)

# === Cell 9: إنشاء تقرير نهائي ===
print("\n📄 إنشاء التقرير النهائي...")

final_report = f"""
# تقرير نموذج تحليل المشاعر العربية

## 📊 معلومات عامة
- **النموذج الأساسي**: {MODEL_NAME}
- **تاريخ التدريب**: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")}
- **مدة التدريب**: {training_info['train_runtime']:.2f} ثانية
- **عدد العينات**:
  - التدريب: {training_info['training_samples']:,}
  - التحقق: {training_info['validation_samples']:,}
  - الاختبار: {training_info['test_samples']:,}

## 📈 الأداء على مجموعة الاختبار
- **الدقة الإجمالية**: {test_metrics['test_accuracy']:.4f}
- **F1-Score (Macro)**: {test_metrics['test_macro_f1']:.4f}
- **F1-Score (Weighted)**: {test_metrics['test_weighted_f1']:.4f}

## 🎭 الأداء حسب الفئة
"""

for label_name in target_names:
    metrics = test_metrics['per_class_metrics'][label_name]
    final_report += f"\n### {label_name}:\n"
    final_report += f"- Precision: {metrics['precision']:.4f}\n"
    final_report += f"- Recall: {metrics['recall']:.4f}\n"
    final_report += f"- F1-Score: {metrics['f1-score']:.4f}\n"
    final_report += f"- عدد العينات: {metrics['support']}\n"

final_report += f"""
## 🔍 تحليل الأخطاء
- **معدل الخطأ**: {error_analysis['error_rate']:.2%}
- **إجمالي الأخطاء**: {error_analysis['total_errors']}

### أنواع الأخطاء الرئيسية:
"""

for error_type, count in sorted(error_analysis['error_types'].items(), key=lambda x: x[1], reverse=True)[:5]:
    percentage = count / error_analysis['total_errors'] * 100
    final_report += f"- {error_type}: {count} ({percentage:.1f}%)\n"

final_report += f"""
## 💾 الملفات المحفوظة
1. **النموذج**: `pytorch_model.bin`
2. **Tokenizer**: `tokenizer_config.json`, `special_tokens_map.json`, `vocab.txt`
3. **معلومات التدريب**: `training_info.json`
4. **تقرير التقييم**: `evaluation_report.txt`
5. **مصفوفة الالتباس**: `confusion_matrix.png`
6. **مقاييس الأداء**: `performance_metrics.png`
7. **أمثلة الاختبار**: `test_examples.json`
8. **تحليل الأخطاء**: `error_analysis.json`

## 🚀 استخدام النموذج

```python
from transformers import pipeline

# تحميل النموذج
classifier = pipeline(
    "text-classification",
    model="{OUT_DIR}",
    tokenizer="{OUT_DIR}"
)

# التنبؤ
result = classifier("النص المراد تحليله")
print(result)
```

## 📝 ملاحظات التحسين
1. تم موازنة البيانات لتحسين الأداء على جميع الفئات
2. تم استخدام تقنيات تنظيم (dropout, label smoothing) لتحسين التعميم
3. تم تحسين معالجة النصوص مع الحفاظ على الإيموجيز والرموز التعبيرية
4. تم استخدام dynamic padding لتحسين كفاءة التدريب
5. تم تقليل MAX_LEN إلى 100 بناءً على تحليل البيانات
"""

with open(os.path.join(OUT_DIR, "final_report.md"), "w", encoding="utf-8") as f:
    f.write(final_report)

print("✅ تم إنشاء التقرير النهائي")

# === Cell 10: ملخص النتائج ===
print("\n" + "="*80)
print("🎉 اكتمل تدريب وتقييم نموذج تحليل المشاعر المحسّن بنجاح!")
print("="*80)

print(f"\n📊 ملخص النتائج:")
print(f"   • الدقة النهائية: {test_metrics['test_accuracy']:.2%}")
print(f"   • F1-Score (Macro): {test_metrics['test_macro_f1']:.4f}")
print(f"   • معدل الخطأ: {error_analysis['error_rate']:.2%}")

print(f"\n📁 موقع النموذج: {OUT_DIR}")

print(f"\n💡 التحسينات المطبقة:")
print(f"   ✓ موازنة البيانات لمعالجة التحيز")
print(f"   ✓ تحسين معالجة النصوص العربية")
print(f"   ✓ استخدام dynamic padding")
print(f"   ✓ تطبيق تقنيات التنظيم المتقدمة")
print(f"   ✓ ضبط hyperparameters بناءً على تحليل البيانات")

print(f"\n🔄 الخطوات التالية المقترحة:")
print(f"   1. تجربة نماذج أخرى (CAMeL-BERT, AraBERT)")
print(f"   2. تطبيق Data Augmentation")
print(f"   3. استخدام Ensemble Methods")
print(f"   4. تحسين معالجة النصوص الطويلة")

print("\n✨ النموذج جاهز للاستخدام!")
