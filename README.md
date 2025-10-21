# Text-Data-Analysis-YouTube-Case-Study
_Analyzing how text features influence engagement on YouTube_

---

## 📘 Overview

**Text Data Analysis: YouTube Case Study** explores how language patterns in YouTube video titles, descriptions, and tags affect viewer engagement. Using Python and data analysis techniques, this project investigates how word choice, length, and sentiment correlate with performance metrics such as views, likes, and comments.

The work is presented in a Jupyter Notebook (`Youtube_data_analyst.ipynb`) as part of an applied data analytics portfolio project.

---
## Importing Required libariries and CSV file.
```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as p

comments = pd.read_csv(
    r"C:\Users\rocke\Downloads\Youtube_Data_Analysis\Dataset\UScomments.csv",
    on_bad_lines="skip"
)
```
## Finding and removing missing values
```python
comments.isnull().sum()
comments.dropna(inplace=True)
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()
sentimen_scores=[]
for comment in comments["comment_text"]:
    score = sia.polarity_scores(str(comment))['compound']
    sentimen_scores.append(score)
sample_df = comments[0:10000]
comments["polarity"] = sentimen_scores
```

```
## 📦 Downloading VADER Lexicon for Sentiment Analysis & 🧠 Importing VADER Sentiment Analyzer and Accessing Comment Text
```python
import nltk
nltk.download("vader_lexicon")


---
## 🎯 Objectives

- Clean and preprocess text-based YouTube metadata
- Perform **Exploratory Data Analysis (EDA)** on engagement metrics
- Extract and analyze **keywords, n-grams, and sentiment**
- Visualize patterns between text features and audience response
- Derive insights for improving video performance and SEO strategy

---

Visualizations include:

- Word Frequency and Word Cloud
- Sentiment Distribution of Titles
- Likes vs Views by Sentiment Polarity
- Average Engagement by Keyword Group
- Upload Frequency Over Time

---

## 🧰 Technologies Used

- **Python 3**
- **Jupyter Notebook**
- **Pandas** – Data manipulation
- **NumPy** – Numerical computing
- **Matplotlib** & **Seaborn** – Data visualization
- **NLTK / TextBlob** – Sentiment analysis and tokenization
- **WordCloud** – Visual representation of common words

## 📦 Dataset Access

The full dataset for this project is **not stored in this repository** to keep the repo lightweight and within GitHub size limits.

If you need access to the datasets for replication or further analysis, **email me**:

**Contact:** paldenlama2708@gmail.com
