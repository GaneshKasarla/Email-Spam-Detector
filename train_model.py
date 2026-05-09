import pickle
import pandas as pd
import string
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# =========================
# LOAD DATASET
# =========================

data = pd.read_csv(
    "SMSSpamCollection",
    sep='\t',
    names=["label", "message"]
)

# Convert labels
data['label'] = data['label'].map({
    'ham': 0,
    'spam': 1
})

# =========================
# CLEAN TEXT
# =========================

def clean_text(text):

    text = text.lower()

    text = ''.join(
        char for char in text
        if char not in string.punctuation
    )

    text = re.sub(r'\s+', ' ', text).strip()

    return text

# Apply cleaning
data['message'] = data['message'].apply(clean_text)

# =========================
# VECTORIZATION
# =========================
vectorizer = TfidfVectorizer(
    stop_words='english',
    ngram_range=(1,2)
)


X = vectorizer.fit_transform(data['message'])

y = data['label']

# =========================
# TRAIN TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =========================
# MODEL TRAINING
# =========================

model = MultinomialNB()

model.fit(X_train, y_train)

pickle.dump(model, open('model.pkl', 'wb'))
pickle.dump(vectorizer, open('vectorizer.pkl', 'wb'))

