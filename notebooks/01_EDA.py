import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
import wandb
import os

# Initialize wandb
wandb.init(project="hisemotions_2026", job_type="eda")

# Ensure the data directory exists and the raw file is present
data_path = "data/raw/train.csv"
images_path = "../images/eda"
if not os.path.exists(data_path):
    print(f"Error: {data_path} not found. Please ensure the data exists in the project root.")
    wandb.finish()
    exit()

# Load data
train_df = pd.read_csv(data_path)

# 1. Label Distribution
labels = ['anger', 'fear', 'joy', 'sadness', 'surprise', 'hope']
class_counts = train_df[labels].sum().sort_values(ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x=class_counts.index, y=class_counts.values)
plt.title("Emotion Class Distribution in Training Data")
plt.savefig("class_distribution.png")
wandb.log({"class_distribution": wandb.Image("class_distribution.png")})

# 2. Text Length analysis
train_df['text_length'] = train_df['text'].apply(lambda x: len(str(x).split()))
plt.figure(figsize=(10, 6))
sns.histplot(train_df['text_length'], bins=50)
plt.title("Text Length (Words) Distribution")
plt.savefig("text_length.png")
wandb.log({"text_length_distribution": wandb.Image("text_length.png")})

# 3. N-gram and Lexical Density Analysis
# Early Modern Spanish has specific vocabulary structures
vectorizer = CountVectorizer(ngram_range=(1, 2), max_features=10)
word_counts = vectorizer.fit_transform(train_df['text'].dropna())
vocab = vectorizer.get_feature_names_out()
counts = word_counts.sum(axis=0).A1

plt.figure(figsize=(12, 6))
sns.barplot(x=vocab, y=counts)
plt.title("Top 10 Uni/Bi-grams in Early Modern Spanish (Uncleaned)")
plt.xticks(rotation=45)
plt.savefig("ngrams.png")
wandb.log({"top_ngrams": wandb.Image("ngrams.png")})

# 4. Co-occurrence Heatmap of Emotions
plt.figure(figsize=(8, 6))
correlation_matrix = train_df[labels].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title("Emotion Co-occurrence Correlation")
plt.savefig("emotion_correlation.png")
wandb.log({"emotion_correlation": wandb.Image("emotion_correlation.png")})

# 5. Log Raw Data Sample
wandb.log({"raw_data_sample": wandb.Table(dataframe=train_df.head(100))})
wandb.finish()
