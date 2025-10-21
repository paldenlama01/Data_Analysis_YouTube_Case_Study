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

```
## 📦 Downloading VADER Lexicon for Sentiment Analysis & 🧠 Importing VADER Sentiment Analyzer and Accessing Comment Text
```python
import nltk
nltk.download("vader_lexicon")

from nltk.sentiment.vader import SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()
sentimen_scores=[]

for comment in comments["comment_text"]:
    score = sia.polarity_scores(str(comment))['compound']
    sentimen_scores.append(score)
sample_df = comments[0:10000]
comments["polarity"] = sentimen_scores

 ##Positive sentiment
! pip install wordcloud

filter_pos = (comments["polarity"] >= 0.8) & ((comments["polarity"] <=1.0))
comments_positive = comments[filter_pos]
total_positive_comments = ' '.join(comments_positive["comment_text"])

from wordcloud import WordCloud, STOPWORDS
wordcloud_positive = WordCloud(stopwords= set(STOPWORDS)).generate(total_positive_comments)
plt.imshow(wordcloud_positive)
plt.axis("off")

 ##Negative sentiment
filter_neg = (comments["polarity"] >= -1.0) & ((comments["polarity"] <=-0.8))
comments_negative = comments[filter_neg]
total_negative_comments = ' '.join(comments_negative["comment_text"])
wordcloud_negative = WordCloud(stopwords= set(STOPWORDS)).generate(total_negative_comments)
plt.imshow(wordcloud_negative)
plt.axis("off")
```
## Perform Emoji Analysis
```python
pip install emoji
import emoji
import emoji

[item["emoji"] for item in emoji_info]
all_emoji_found = []

for comment in comments["comment_text"]:
    emoji_info = emoji.emoji_list(comment)
    emoji_found = [item["emoji"] for item in emoji_info]
    all_emoji_found.extend(emoji_found)

from collections import Counter 
emoji_count_list_top10 = Counter(all_emoji_found).most_common(10)
emojis = [emoji for emoji, count in emoji_count_list_top10]
counts = [count for emoji, count in emoji_count_list_top10]

pip install plotly
import plotly.graph_objs as go
from plotly.offline import iplot

iplot([go.Bar(x = emojis , y = counts)])
```
## Collect entire youtube data collection 
```python
import os

files = os.listdir(r'C:\Users\rocke\Downloads\Youtube_Data_Analysis\Dataset\additional_data')
files_csv = [file for file in files if '.csv' in file]

import warnings
from warnings import filterwarnings
filterwarnings('ignore')

import pandas as pd
full_df = pd.DataFrame()
path = r'C:\Users\rocke\Downloads\Youtube_Data_Analysis\Dataset\additional_data'

for file in files_csv:
    current_df = pd.read_csv(os.path.join(path, file), encoding='iso-8859-1', on_bad_lines="skip")
    full_df = pd.concat([full_df, current_df], ignore_index=True)
```
## Analyzing the most liked category
```python
full_df['category_id'].unique()
json_df = pd.read_json(r'C:\Users\rocke\Downloads\Youtube_Data_Analysis\Dataset\additional_data/US_category_id.json')
cat_dict = {}

for item in json_df['items'].values:
    cat_dict[int(item['id'])] = item['snippet']['title']
full_df['category_name'] = full_df['category_id'].map(cat_dict)

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(12,8))
sns.boxplot(x='category_name', y='likes', data=full_df)
plt.xticks(rotation='vertical')
```
## Analyzing Youtube Audience Engagement
```python
full_df['like_rate'] = full_df['likes']/full_df['views']*100
full_df['dislike_rate'] = full_df['dislikes']/full_df['views']*100
full_df['comment_count_rate'] = full_df['comment_count']/full_df['views']*100

plt.figure(figsize=(8,6))
sns.boxplot(x='category_name', y='like_rate', data=full_df)
plt.xticks(rotation='vertical')
plt.show()

sns.regplot(x='views', y='likes', data=full_df)
plt.show()

full_df[['views','likes','dislikes']]
full_df[['views','likes','dislikes']].corr()
sns.heatmap(full_df[['views','likes','dislikes']].corr(),annot=True)
```
# Analyzing Trending Youtube Videos by Channel 
```python
full_df['channel_title'].value_counts()
cdf = full_df.groupby(['channel_title']).size().sort_values(ascending=False).reset_index()
cdf.columns=['channel_title', 'total_videos']

import plotly.express as px
fig = px.bar(cdf[:20], x='channel_title', y='total_videos',title="Top 20 Channels by Number of Videos")
fig.show()
```
## Does punctuation have an impact on views, likes and dislikes? 
```python
import string
string.punctuation
len([char for char in full_df['title'][0] if char in string.punctuation])
def punc_count(text):
    return len([char for char in text if char in string.punctuation])

sample= full_df[0:10000]
sample['count_punc'] = sample['title'].apply(punc_count)
plt.figure(figsize=(8,6))
sns.boxplot(x='count_punc', y='views', data=sample)
plt.title("Relationship Between Punctuation Frequency and Views")
plt.show()

plt.figure(figsize=(8,6))
sns.boxplot(x='count_punc', y='likes', data=sample)
plt.title("Relationship Between Punctuation Frequency and Likes")
plt.show()

```


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
