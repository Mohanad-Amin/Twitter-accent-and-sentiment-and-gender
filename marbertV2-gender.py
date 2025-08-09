# تدريب نموذج MarBERTv2 لتصنيف الجنس (ثلاث فئات)
# يعمل هذا الكود على Google Colab مع GPU A100

# ========================
# 1. تثبيت المكتبات المطلوبة
# ========================
!pip -q install datasets

# ========================
# 2. استيراد المكتبات
# ========================
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    DataCollatorWithPadding
)
from datasets import Dataset as HFDataset
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ========================
# 3. تحميل وتحضير البيانات
# ========================
print("جاري تحميل البيانات...")
# قراءة ملف Excel
df = pd.read_excel('/content/drive/MyDrive/Tweet_Project/Data/Gender_dallah_labeled0.xlsx')

# عرض معلومات أساسية عن البيانات
print(f"عدد الصفوف الكلي: {len(df)}")
print(f"الأعمدة: {df.columns.tolist()}")
print(f"\nتوزيع الجنس:")
print(df['gender'].value_counts())

# تنظيف البيانات - الآن نحتفظ بجميع الفئات الثلاث
# تأكد من وجود القيم المطلوبة
valid_genders = ['male', 'female', 'unknown']
df_clean = df[df['gender'].isin(valid_genders)].copy()

# إذا كانت هناك قيم أخرى، يمكن تحويلها إلى unknown
# df.loc[~df['gender'].isin(['male', 'female']), 'gender'] = 'unknown'

print(f"\nعدد الصفوف بعد التنظيف: {len(df_clean)}")
print(f"توزيع الجنس بعد التنظيف:")
print(df_clean['gender'].value_counts())

# دمج المعلومات النصية (الاسم + الوصف)
df_clean['text'] = df_clean['author.name'].fillna('') + ' ' + df_clean['author.description'].fillna('')
df_clean['text'] = df_clean['text'].str.strip()

# تحويل التسميات إلى أرقام (الآن ثلاث فئات)
label_map = {'male': 0, 'female': 1, 'unknown': 2}
df_clean['label'] = df_clean['gender'].map(label_map)

# التحقق من توزيع التسميات
print(f"\nتوزيع التسميات الرقمية:")
print(df_clean['label'].value_counts().sort_index())

# ========================
# 4. تقسيم البيانات
# ========================
# تقسيم البيانات إلى تدريب واختبار وتحقق
X = df_clean['text'].values
y = df_clean['label'].values

# تقسيم أولي: 80% تدريب+تحقق، 20% اختبار
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# تقسيم ثانوي: 80% تدريب، 20% تحقق من البيانات المتبقية
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.2, random_state=42, stratify=y_temp
)

print(f"\nأحجام مجموعات البيانات:")
print(f"التدريب: {len(X_train)}")
print(f"التحقق: {len(X_val)}")
print(f"الاختبار: {len(X_test)}")

# عرض توزيع الفئات في كل مجموعة
for name, labels in [("التدريب", y_train), ("التحقق", y_val), ("الاختبار", y_test)]:
    unique, counts = np.unique(labels, return_counts=True)
    print(f"\nتوزيع الفئات في مجموعة {name}:")
    for label, count in zip(unique, counts):
        gender_name = ['Male', 'Female', 'Unknown'][label]
        print(f"  {gender_name}: {count}")

# ========================
# 5. إعداد النموذج والمحلل اللغوي
# ========================
print("\nجاري تحميل نموذج MarBERTv2...")
model_name = "UBC-NLP/MARBERTv2"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# تحديث النموذج ليدعم 3 فئات
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=3,  # تغيير من 2 إلى 3
    id2label={0: "male", 1: "female", 2: "unknown"},
    label2id={"male": 0, "female": 1, "unknown": 2}
)

# ========================
# 6. إعداد البيانات للتدريب
# ========================
def prepare_dataset(texts, labels):
    """تحضير البيانات بصيغة مناسبة للتدريب"""
    dataset_dict = {
        'text': texts,
        'label': labels
    }
    return HFDataset.from_dict(dataset_dict)

# إنشاء مجموعات البيانات
train_dataset = prepare_dataset(X_train, y_train)
val_dataset = prepare_dataset(X_val, y_val)
test_dataset = prepare_dataset(X_test, y_test)

# دالة لتحليل النصوص
def tokenize_function(examples):
    return tokenizer(
        examples['text'],
        padding='max_length',
        truncation=True,
        max_length=256
    )

# تطبيق التحليل اللغوي
print("\nجاري معالجة النصوص...")
train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

# إعداد صيغة البيانات
train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

# ========================
# 7. دالة حساب المقاييس
# ========================
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='weighted'
    )

    # حساب المقاييس لكل فئة على حدة
    precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
        labels, predictions, average=None, labels=[0, 1, 2]
    )

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'precision_male': precision_per_class[0],
        'precision_female': precision_per_class[1],
        'precision_unknown': precision_per_class[2],
        'recall_male': recall_per_class[0],
        'recall_female': recall_per_class[1],
        'recall_unknown': recall_per_class[2],
        'f1_male': f1_per_class[0],
        'f1_female': f1_per_class[1],
        'f1_unknown': f1_per_class[2],
    }

# ========================
# 8. معالجة عدم توازن الفئات (إضافة أوزان للفئات)
# ========================
from sklearn.utils.class_weight import compute_class_weight

# حساب أوزان الفئات
classes = np.array([0, 1, 2])
class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}

print(f"\nأوزان الفئات:")
for class_id, weight in class_weight_dict.items():
    class_name = ['Male', 'Female', 'Unknown'][class_id]
    print(f"  {class_name}: {weight:.3f}")

# إنشاء دالة loss مخصصة مع الأوزان
class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        # تحويل أوزان الفئات إلى tensor
        device = logits.device
        weights = torch.tensor([class_weight_dict[i] for i in range(3)],
                              dtype=torch.float32, device=device)

        # استخدام CrossEntropyLoss مع الأوزان
        loss_fct = torch.nn.CrossEntropyLoss(weight=weights)
        loss = loss_fct(logits.view(-1, 3), labels.view(-1))

        return (loss, outputs) if return_outputs else loss

# ========================
# 9. إعدادات التدريب
# ========================
training_args = TrainingArguments(
    output_dir='./marbert_gender_classifier_3class',
    eval_strategy="steps",
    eval_steps=300,
    save_strategy="steps",
    save_steps=600,
    learning_rate=2e-5,
    per_device_train_batch_size=24,  # تقليل قليلاً بسبب وجود فئة إضافية
    per_device_eval_batch_size=48,
    num_train_epochs=6,  # زيادة عدد العصور قليلاً
    weight_decay=0.01,
    warmup_steps=500,
    logging_dir='./logs',
    logging_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model='f1',
    greater_is_better=True,
    save_total_limit=3,
    fp16=True,
    gradient_checkpointing=True,
    gradient_accumulation_steps=2,
    report_to="none",
    seed=42,
)

# إنشاء مُجمِّع البيانات
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# ========================
# 10. إنشاء المدرب وبدء التدريب
# ========================
trainer = WeightedTrainer(  # استخدام المدرب المخصص
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)

print("\nبدء التدريب...")
train_result = trainer.train()

# حفظ النموذج النهائي
print("\nحفظ النموذج...")
trainer.save_model('./marbert_gender_classifier_3class_final')
tokenizer.save_pretrained('./marbert_gender_classifier_3class_final')

# ========================
# 11. التقييم على مجموعة الاختبار
# ========================
print("\nتقييم النموذج على مجموعة الاختبار...")
test_results = trainer.evaluate(eval_dataset=test_dataset)

print("\nنتائج الاختبار:")
for key, value in test_results.items():
    if key.startswith('eval_'):
        metric_name = key.replace('eval_', '')
        print(f"{metric_name}: {value:.4f}")

# ========================
# 12. التنبؤ وتحليل النتائج
# ========================
print("\nإنشاء التنبؤات...")
predictions = trainer.predict(test_dataset)
y_pred = np.argmax(predictions.predictions, axis=1)

# مصفوفة الخلط للفئات الثلاث
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Male', 'Female', 'Unknown'],
            yticklabels=['Male', 'Female', 'Unknown'])
plt.title('Confusion Matrix - Gender Classification (3 Classes)')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig('confusion_matrix_3class.png', dpi=300, bbox_inches='tight')
plt.show()

# تقرير التصنيف المفصل للفئات الثلاث
from sklearn.metrics import classification_report
report = classification_report(y_test, y_pred,
                             target_names=['Male', 'Female', 'Unknown'],
                             output_dict=True)

print("\nتقرير التصنيف المفصل:")
print(f"{'Class':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}")
print("-" * 60)
for label in ['Male', 'Female', 'Unknown']:
    metrics = report[label]
    print(f"{label:<10} {metrics['precision']:<10.4f} {metrics['recall']:<10.4f} "
          f"{metrics['f1-score']:<10.4f} {int(metrics['support']):<10}")

# عرض المقاييس الإجمالية
print(f"\n{'Macro Avg':<10} {report['macro avg']['precision']:<10.4f} "
      f"{report['macro avg']['recall']:<10.4f} {report['macro avg']['f1-score']:<10.4f}")
print(f"{'Weighted Avg':<10} {report['weighted avg']['precision']:<10.4f} "
      f"{report['weighted avg']['recall']:<10.4f} {report['weighted avg']['f1-score']:<10.4f}")

# ========================
# 13. حفظ النتائج
# ========================
# حفظ النتائج في ملف
results_df = pd.DataFrame({
    'text': X_test,
    'true_label': y_test,
    'predicted_label': y_pred,
    'true_gender': ['male' if l == 0 else 'female' if l == 1 else 'unknown' for l in y_test],
    'predicted_gender': ['male' if l == 0 else 'female' if l == 1 else 'unknown' for l in y_pred]
})
results_df.to_csv('test_predictions_3class.csv', index=False)

# ========================
# 14. دالة للتنبؤ بنصوص جديدة (محدثة للفئات الثلاث)
# ========================
def predict_gender(text, model, tokenizer):
    """
    دالة للتنبؤ بجنس المؤلف من النص (ثلاث فئات)
    """
    # تحليل النص
    inputs = tokenizer(text, return_tensors="pt", truncation=True,
                      padding=True, max_length=256)

    # نقل البيانات إلى GPU إن وجد
    device = model.device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # التنبؤ
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_class = torch.argmax(predictions, dim=-1).item()
        confidence = predictions[0][predicted_class].item()

    gender_map = {0: 'male', 1: 'female', 2: 'unknown'}
    gender = gender_map[predicted_class]

    # إرجاع الاحتماليات لجميع الفئات
    probabilities = {
        'male': predictions[0][0].item(),
        'female': predictions[0][1].item(),
        'unknown': predictions[0][2].item()
    }

    return gender, confidence, probabilities

# ========================
# 15. تحليل الأخطاء
# ========================
def analyze_errors(y_true, y_pred, texts):
    """تحليل الأخطاء في التصنيف"""
    errors = []
    for i, (true_label, pred_label, text) in enumerate(zip(y_true, y_pred, texts)):
        if true_label != pred_label:
            errors.append({
                'index': i,
                'text': text,
                'true_label': ['male', 'female', 'unknown'][true_label],
                'predicted_label': ['male', 'female', 'unknown'][pred_label],
                'error_type': f"{['male', 'female', 'unknown'][true_label]} -> {['male', 'female', 'unknown'][pred_label]}"
            })
    return errors

# تحليل الأخطاء
errors = analyze_errors(y_test, y_pred, X_test)
print(f"\nعدد الأخطاء: {len(errors)} من أصل {len(y_test)} ({len(errors)/len(y_test)*100:.1f}%)")

# عرض أنواع الأخطاء الأكثر شيوعاً
error_types = {}
for error in errors:
    error_type = error['error_type']
    error_types[error_type] = error_types.get(error_type, 0) + 1

print("\nأنواع الأخطاء الأكثر شيوعاً:")
for error_type, count in sorted(error_types.items(), key=lambda x: x[1], reverse=True):
    print(f"  {error_type}: {count} مرة")

# ========================
# 16. أمثلة على الاستخدام
# ========================
print("\n" + "="*60)
print("أمثلة على التنبؤات:")
print("="*60)

test_examples = [
    "د. أحمد محمد - أستاذ في كلية الهندسة",
    "سارة العلي - مصممة جرافيك ومهتمة بالفنون",
    "مهندس برمجيات في شركة تقنية",
    "محب للقراءة والسفر",
    "Account - Business - Tech",
    ""  # نص فارغ
]

for text in test_examples:
    if not text.strip():
        text = "[نص فارغ]"
    gender, confidence, probabilities = predict_gender(text, model, tokenizer)
    print(f"\nالنص: {text}")
    print(f"الجنس المتوقع: {gender} (الثقة: {confidence:.2%})")
    print(f"احتماليات جميع الفئات:")
    for class_name, prob in probabilities.items():
        print(f"  {class_name}: {prob:.2%}")

print("\n" + "="*60)
print("انتهى التدريب والتقييم بنجاح!")
print("="*60)
