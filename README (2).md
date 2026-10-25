# Emotion Bloom

Classify a sentence into one of six emotions (sadness, joy, love, anger, fear, surprise) and see the result as a flower whose petals grow with each emotion's probability.

The project has two parts:

- `nlpproj_py.ipynb`: trains and compares three text classifiers.
- `emotion-reader-v2.html`: a single-file interface for trying them.

## Models

Trained on the 16,000-sentence emotion dataset (`train.txt`, 80/20 split, `random_state=42`).

| Model | Features | Test accuracy |
| --- | --- | --- |
| Logistic Regression | TF-IDF | 86.3% |
| Naive Bayes | Word counts | 76.8% |
| Naive Bayes | TF-IDF | 66.1% |

## Preprocessing

Every sentence goes through the same steps before it reaches a model:

1. Lowercase
2. Remove punctuation
3. Remove digits
4. Remove non-ASCII characters (including emojis)
5. Remove English stopwords (NLTK list)

The interface shows these steps live under "What the models actually read".

## Run the interface

Open `emotion-reader-v2.html` in any browser. With no backend it runs in preview mode, scoring sentences with a small built-in word list. That is only for trying the design; the results are not from your trained models.

## Connect your trained models

**1. Save the models.** Add this to the end of the notebook:

```python
import joblib
joblib.dump(emotion_numbers, "labels.pkl")
joblib.dump(stop_words, "stopwords.pkl")
joblib.dump(bow_vectorizer, "bow.pkl");     joblib.dump(nb_model, "nb.pkl")
joblib.dump(tfidf_vectorizer, "tfidf.pkl"); joblib.dump(nb2_model, "nbt.pkl")
joblib.dump(logistic_model, "lr.pkl")
```

**2. Create `app.py`** next to those files:

```python
import string, joblib
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

labels = joblib.load("labels.pkl")            # {"sadness": 0, ...}
names = {v: k for k, v in labels.items()}
stop = joblib.load("stopwords.pkl")
bow, tfidf = joblib.load("bow.pkl"), joblib.load("tfidf.pkl")
models = {
    "lr":  (tfidf, joblib.load("lr.pkl")),
    "nb":  (bow,   joblib.load("nb.pkl")),
    "nbt": (tfidf, joblib.load("nbt.pkl")),
}

def clean(t):
    t = t.lower().translate(str.maketrans("", "", string.punctuation))
    t = "".join(c for c in t if not c.isdigit() and c.isascii())
    return " ".join(w for w in t.split() if w not in stop)

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class Req(BaseModel):
    text: str
    model: str = "lr"

@app.post("/predict")
def predict(r: Req):
    vec, clf = models[r.model]
    p = clf.predict_proba(vec.transform([clean(r.text)]))[0]
    return {"probs": {names[int(c)]: float(x) for c, x in zip(clf.classes_, p)}}
```

**3. Start it:**

```bash
pip install fastapi uvicorn scikit-learn joblib
uvicorn app:app --port 8000
```

**4. Point the page at it.** In `emotion-reader-v2.html`, set:

```js
const API_URL = "http://localhost:8000/predict";
```

Open the file locally. The preview note disappears and every result comes from your models. The hosted preview link cannot reach `localhost`, so use the downloaded file.

## API

`POST /predict`

Request: `{"text": "i feel so hopeful", "model": "lr"}` where `model` is `lr`, `nb` or `nbt`.

Response: `{"probs": {"sadness": 0.04, "joy": 0.71, "love": 0.09, "anger": 0.03, "fear": 0.05, "surprise": 0.08}}`

## Notes

- Sentences are short and informal, so the models do best on text that looks like the training data (first-person statements starting with "i feel").
- The TF-IDF Naive Bayes model is the weakest here; it tends to favour the most common classes. It is kept in the comparison on purpose.
