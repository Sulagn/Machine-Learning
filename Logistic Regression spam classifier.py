# ============================================================
#  Project: Spam Email Classifier — Logistic Regression
#  Goal   : Learn Logistic Regression by building a spam
#           detector using word-frequency features.
#  Dataset: Synthetically generated (no download needed)
# ============================================================

# ---------- 0. Install dependencies (run once) --------------
# pip install numpy pandas matplotlib scikit-learn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_curve, auc,
    classification_report
)

# ============================================================
# STEP 1 — Generate a Synthetic Email Dataset
# ============================================================
# We create realistic fake emails so zero downloads are needed.
# Spam emails are loaded with trigger words; ham emails are normal.

np.random.seed(42)

SPAM_PHRASES = [
    "Congratulations! You have won a $1000 gift card. Click here now!",
    "FREE offer! Claim your prize today. Limited time only!",
    "You are selected for a cash reward. Verify your account immediately.",
    "URGENT: Your account will be suspended. Click the link to confirm.",
    "Make money fast! Work from home, earn $5000 weekly. No experience needed.",
    "Buy cheap medications online. No prescription required. Lowest prices!",
    "You have been chosen. Claim your free iPhone now. Act fast!",
    "Hot singles in your area want to meet you tonight. Click here!",
    "Earn unlimited income online. Secret method revealed. Sign up free!",
    "WINNER! Your email has been selected. Collect your lottery prize now.",
    "Lose 20 pounds in 2 weeks! Miracle pill doctors hate. Order now!",
    "Your PayPal account is limited. Verify now to restore full access.",
    "Double your Bitcoin in 24 hours. Guaranteed returns. Invest now!",
    "Cheap Rolex watches. 90% off luxury brands. Shop now!",
    "You owe back taxes. Pay immediately or face legal action. Call now!",
    "Free credit check. Improve your score overnight. No cost to you!",
    "Exclusive deal for you only! Buy now and save 80%. Today only!",
    "Nigerian prince needs your help. $10 million waiting. Reply urgently.",
    "Prescription drugs at lowest cost. No doctor needed. Order online!",
    "Get rich quick scheme revealed. Thousands made $10k last month.",
]

HAM_PHRASES = [
    "Hey, are we still on for lunch tomorrow? Let me know what time works.",
    "Please find attached the quarterly report for your review.",
    "Hi team, the meeting has been moved to 3pm on Thursday. Please update your calendars.",
    "Thanks for your help yesterday. The project is looking great!",
    "Can you send me the updated budget spreadsheet when you get a chance?",
    "Reminder: your dentist appointment is on Friday at 10am.",
    "Happy birthday! Hope you have a wonderful day.",
    "I have reviewed the contract and have a few questions. Can we schedule a call?",
    "The package you ordered has been shipped and will arrive by Wednesday.",
    "Just checking in to see how the onboarding process is going.",
    "Please review the attached proposal and share your feedback by end of week.",
    "The client meeting went well. They approved the new design direction.",
    "Your flight booking confirmation: Mumbai to Delhi, 15th, 9am departure.",
    "Hi, I wanted to follow up on our conversation from last week.",
    "The server maintenance is scheduled for Sunday between 2am and 4am.",
    "Could you help me understand the new expense policy? I have a few questions.",
    "We are planning a team outing next month. Please fill out the availability form.",
    "The library book you reserved is now available for pickup.",
    "Great presentation today! The data visualisations were really clear.",
    "Your invoice has been processed and payment will be made within 5 business days.",
]

# Expand to 500 samples with slight variations
def augment(phrases, label, n=250):
    emails, labels = [], []
    fillers = ["FYI", "Note:", "Update:", "Info:", "Alert:"]
    for i in range(n):
        base = phrases[i % len(phrases)]
        # small variation: prepend a filler word occasionally
        if i % 7 == 0:
            base = np.random.choice(fillers) + " " + base
        emails.append(base)
        labels.append(label)
    return emails, labels

spam_emails, spam_labels = augment(SPAM_PHRASES, label=1, n=250)
ham_emails,  ham_labels  = augment(HAM_PHRASES,  label=0, n=250)

emails = spam_emails + ham_emails
labels = spam_labels + ham_labels

df = pd.DataFrame({"email": emails, "label": labels})
df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle

print("=" * 60)
print("STEP 1 — Dataset Preview")
print("=" * 60)
print(f"Total emails : {len(df)}")
print(f"Spam (1)     : {df['label'].sum()}")
print(f"Ham  (0)     : {(df['label'] == 0).sum()}")
print("\nSample emails:")
for _, row in df.sample(4, random_state=1).iterrows():
    tag = "🚨 SPAM" if row.label == 1 else "✅ HAM "
    print(f"  [{tag}] {row.email[:70]}...")


# ============================================================
# STEP 2 — Convert Text → Numbers with TF-IDF
# ============================================================
# Logistic Regression needs numbers, not text.
# TF-IDF converts each email into a vector of word importance scores.
#   TF  = Term Frequency   (how often a word appears in this email)
#   IDF = Inverse Document Frequency (how rare/unique the word is overall)
# Rare but present words get a HIGH score → great signal for spam detection.

vectorizer = TfidfVectorizer(
    stop_words="english",   # ignore "the", "is", "and", etc.
    max_features=500,       # keep only top 500 words
    ngram_range=(1, 2)      # single words AND 2-word phrases ("click here")
)

X = vectorizer.fit_transform(df["email"])
y = df["label"].values

print("\n" + "=" * 60)
print("STEP 2 — TF-IDF Feature Matrix")
print("=" * 60)
print(f"Shape: {X.shape[0]} emails × {X.shape[1]} features (words/phrases)")
print("Each email is now a row of TF-IDF scores.")
print("Most common features:", vectorizer.get_feature_names_out()[:10].tolist())


# ============================================================
# STEP 3 — Train / Test Split
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\n" + "=" * 60)
print("STEP 3 — Train / Test Split")
print("=" * 60)
print(f"Training emails : {X_train.shape[0]}")
print(f"Testing  emails : {X_test.shape[0]}")
print(f"Spam % in train : {y_train.mean()*100:.1f}%  (stratified split)")


# ============================================================
# STEP 4 — Train Logistic Regression
# ============================================================
# Logistic Regression outputs a PROBABILITY (0 to 1):
#   P(spam) > 0.5  →  classify as spam
#   P(spam) ≤ 0.5  →  classify as ham
#
# Internally it learns weights for each word:
#   high positive weight → word pushes toward spam
#   high negative weight → word pushes toward ham

model = LogisticRegression(
    C=1.0,          # regularisation strength (smaller C = more regularised)
    max_iter=1000,
    solver="lbfgs",
    random_state=42
)
model.fit(X_train, y_train)

print("\n" + "=" * 60)
print("STEP 4 — Model Trained")
print("=" * 60)
print("Logistic Regression learns a weight for each word.")
print("Top SPAM words (highest positive weights):")
feature_names = vectorizer.get_feature_names_out()
coefs         = model.coef_[0]
top_spam_idx  = np.argsort(coefs)[-10:][::-1]
top_ham_idx   = np.argsort(coefs)[:10]

for idx in top_spam_idx:
    print(f"  +{coefs[idx]:6.3f}  '{feature_names[idx]}'")

print("\nTop HAM words (most negative weights — push away from spam):")
for idx in top_ham_idx:
    print(f"  {coefs[idx]:6.3f}  '{feature_names[idx]}'")


# ============================================================
# STEP 5 — Evaluate the Model
# ============================================================
y_pred      = model.predict(X_test)
y_pred_prob = model.predict_proba(X_test)[:, 1]  # probability of spam

acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec  = recall_score(y_test, y_pred)
f1   = f1_score(y_test, y_pred)

print("\n" + "=" * 60)
print("STEP 5 — Model Evaluation (test set)")
print("=" * 60)
print(f"Accuracy  : {acc:.4f}  ({acc*100:.1f}%)")
print(f"Precision : {prec:.4f}  — of predicted spam, {prec*100:.1f}% really is spam")
print(f"Recall    : {rec:.4f}  — of actual spam, {rec*100:.1f}% was caught")
print(f"F1 Score  : {f1:.4f}  — harmonic mean of Precision & Recall")
print("""
Key trade-off:
  High Precision → fewer false alarms (legit emails marked spam)
  High Recall    → fewer misses (spam that slips through to inbox)
  F1 Score       → balanced measure of both
""")
print("Full Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Ham", "Spam"]))


# ============================================================
# STEP 6 — Predict on New Emails
# ============================================================
new_emails = [
    "Congratulations! You won a free iPhone. Click here to claim now!",
    "Hi Sarah, can you review the slides before tomorrow's presentation?",
    "URGENT: Your bank account has been suspended. Verify now immediately!",
    "The team lunch is at 1pm on Friday. See you there!",
]

new_vectors = vectorizer.transform(new_emails)
new_preds   = model.predict(new_vectors)
new_probs   = model.predict_proba(new_vectors)[:, 1]

print("=" * 60)
print("STEP 6 — Predict New Emails")
print("=" * 60)
for email, pred, prob in zip(new_emails, new_preds, new_probs):
    tag = "🚨 SPAM" if pred == 1 else "✅ HAM "
    print(f"[{tag}] ({prob*100:.1f}% spam)  {email[:65]}...")


# ============================================================
# STEP 7 — Visualisations
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(13, 10))
fig.suptitle("Logistic Regression — Spam Email Classifier", fontsize=16)

# --- Plot 1: Confusion Matrix ---
ax = axes[0, 0]
cm = confusion_matrix(y_test, y_pred)
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
ax.set_xticklabels(["Ham", "Spam"]); ax.set_yticklabels(["Ham", "Spam"])
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
ax.set_title("Confusion Matrix")
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
                fontsize=18, fontweight="bold")
labels_cm = ["True Neg\n(Ham correct)", "False Pos\n(Ham → Spam)", 
             "False Neg\n(Spam missed)", "True Pos\n(Spam caught)"]
positions = [(0,0),(1,0),(0,1),(1,1)]
for label, (xi, yi) in zip(labels_cm, positions):
    ax.text(xi, yi + 0.35, label, ha="center", va="center", fontsize=7, color="gray")

# --- Plot 2: ROC Curve ---
ax = axes[0, 1]
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
roc_auc      = auc(fpr, tpr)
ax.plot(fpr, tpr, color="#378ADD", linewidth=2.5, label=f"ROC Curve (AUC = {roc_auc:.3f})")
ax.plot([0, 1], [0, 1], "r--", linewidth=1.5, label="Random classifier")
ax.fill_between(fpr, tpr, alpha=0.1, color="#378ADD")
ax.set_xlabel("False Positive Rate (Ham → Spam)")
ax.set_ylabel("True Positive Rate (Spam caught)")
ax.set_title("ROC Curve\n(AUC closer to 1.0 = better)")
ax.legend()

# --- Plot 3: Top spam / ham words ---
ax = axes[1, 0]
n = 10
words  = ([feature_names[i] for i in top_spam_idx] +
          [feature_names[i] for i in top_ham_idx])
scores = ([coefs[i] for i in top_spam_idx] +
          [coefs[i] for i in top_ham_idx])
colors = ["#E24B4A"] * n + ["#378ADD"] * n
y_pos  = range(len(words))
bars   = ax.barh(list(y_pos), scores, color=colors, edgecolor="white")
ax.set_yticks(list(y_pos))
ax.set_yticklabels(words, fontsize=9)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlabel("Weight (positive = spam signal, negative = ham signal)")
ax.set_title("Most Influential Words")
spam_patch = mpatches.Patch(color="#E24B4A", label="Spam words")
ham_patch  = mpatches.Patch(color="#378ADD", label="Ham words")
ax.legend(handles=[spam_patch, ham_patch], fontsize=9)

# --- Plot 4: Prediction probability distribution ---
ax = axes[1, 1]
spam_probs = y_pred_prob[y_test == 1]
ham_probs  = y_pred_prob[y_test == 0]
ax.hist(ham_probs,  bins=20, alpha=0.7, color="#378ADD", label="Actual Ham",  edgecolor="white")
ax.hist(spam_probs, bins=20, alpha=0.7, color="#E24B4A", label="Actual Spam", edgecolor="white")
ax.axvline(0.5, color="black", linestyle="--", linewidth=2, label="Decision boundary (0.5)")
ax.set_xlabel("Predicted Probability of Spam")
ax.set_ylabel("Number of Emails")
ax.set_title("Probability Distribution\n(well-separated = confident model)")
ax.legend()

plt.tight_layout()
plt.savefig("spam_classifier_results.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nPlot saved as 'spam_classifier_results.png'")


# ============================================================
# STEP 8 — Key Takeaways
# ============================================================
print("\n" + "=" * 60)
print("KEY TAKEAWAYS")
print("=" * 60)
print("""
1. LOGISTIC REGRESSION outputs a PROBABILITY via the sigmoid:
     P(spam) = 1 / (1 + e^(-z))
   where z = w1*word1 + w2*word2 + ... + bias

2. TF-IDF VECTORISATION turns text into numbers:
   - Words like "free", "click", "winner" get high spam weights
   - Words like "attached", "meeting", "review" get high ham weights

3. THE CONFUSION MATRIX shows 4 outcome types:
   - True Positive  = spam caught (good ✅)
   - True Negative  = ham kept    (good ✅)
   - False Positive = ham blocked (annoying ⚠️)
   - False Negative = spam missed (bad ❌)

4. PRECISION vs RECALL trade-off:
   - Increase threshold (e.g. 0.7) → fewer false alarms, more spam leaks
   - Decrease threshold (e.g. 0.3) → catches more spam, more false alarms

5. ROC-AUC summarises performance across ALL thresholds:
   - 1.0 = perfect, 0.5 = random guessing
   - AUC > 0.95 means excellent separation between spam and ham

6. REGULARISATION (parameter C):
   - Small C → stronger penalty → simpler model (less overfit)
   - Large C → weaker penalty  → complex model (can overfit)
""")