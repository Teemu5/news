import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import logging
import os
import requests
import zipfile
from scipy import sparse
import subprocess
from models import create_user_profile

def update_tfidf_matrix(tfidf, new_articles):
    # This function updates the TF-IDF matrix with new articles
    new_tfidf_matrix = tfidf.transform(new_articles['combined_text'])
    return new_tfidf_matrix

def add_new_articles(news_df, new_articles):
    # Append new articles to the news dataframe
    updated_news_df = news_df.append(new_articles, ignore_index=True)
    updated_news_df['combined_text'] = updated_news_df['Title'].fillna('') + " " + updated_news_df['Abstract'].fillna('')
    return updated_news_df

def recommend_new_articles(tfidf, tfidf_matrix, user_profiles, new_articles):
    # Update the news dataframe and tfidf matrix with new articles
    updated_news_df = add_new_articles(news_df, new_articles)
    new_tfidf_matrix = update_tfidf_matrix(tfidf, new_articles)
    full_tfidf_matrix = sparse.vstack([tfidf_matrix, new_tfidf_matrix])
    
    # Compute recommendations for each user based on cosine similarity
    recommendations = {}
    for user_id, profile in user_profiles.items():
        sim_scores = cosine_similarity(profile, full_tfidf_matrix)
        top_indices = np.argsort(sim_scores[0])[::-1][:10]  # Top 10 recommendations
        recommendations[user_id] = updated_news_df.iloc[top_indices]['NewsID'].values.tolist()
    
    return recommendations
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_and_preprocess_data(news_path, behaviors_path):
    news_df = pd.read_csv(news_path, delimiter='\t', names=['NewsID', 'Category', 'SubCategory', 'Title', 'Abstract', 'URL', 'TitleEntities', 'AbstractEntities'])
    behaviors_df = pd.read_csv(behaviors_path, delimiter='\t', names=['ImpressionID', 'UserID', 'Time', 'History', 'Impressions'])
    
    # Preprocess and combine title and abstract for TF-IDF
    news_df['combined_text'] = news_df['Title'].fillna('') + " " + news_df['Abstract'].fillna('')
    return news_df, behaviors_df
def save_data(df, df_name):
    directory = '/app/data'
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(f"/app/data/{df_name}.pkl", 'wb') as f:
        pickle.dump(df, f)
def save_data_frames(news_df, behaviors_df, user_profiles, tfidf_matrix):
    save_data(news_df, "news_df")
    save_data(user_profiles, "user_profiles")
    save_data(tfidf_matrix, "tfidf_matrix")

def build_user_profiles_and_save_matrix(news_df, behaviors_df, model = "tfidf"):
    user_profiles = {}
    news_id_index = pd.Series(data=news_df.index, index=news_df.NewsID).to_dict()
    if model == "tfidf":
        tfidf = TfidfVectorizer(stop_words='english', max_features=1000)
        tfidf_matrix = tfidf.fit_transform(news_df['combined_text'])
    for _, row in behaviors_df.iterrows():
        user_id = row['UserID']
        clicked_articles = [impression.split("-")[0] for impression in row['Impressions'].split() if impression.endswith("-1")]
        indices = [news_id_index[article_id] for article_id in clicked_articles if article_id in news_id_index]
        if indices:
            if model == "tfidf":
                user_profiles[user_id] = tfidf_matrix[indices].mean(axis=0)
            else:
                user_profiles[user_id] = create_user_profile(clicked_articles, model)
    if model == "tfidf":
        # Save user profiles and the TF-IDF matrix
        save_data_frames(news_df, behaviors_df, user_profiles, tfidf_matrix)
    else:
        save_data(user_profiles, "fastformer_user_profiles")

import feedparser

def fetch_rss_feed(feed_url):
    # Parse the RSS feed
    feed = feedparser.parse(feed_url)
    articles = []
    for entry in feed.entries:
        articles.append({
            'NewsID': entry.id,
            'Title': entry.title,
            'Abstract': entry.summary,
            'URL': entry.link
        })
    return articles

import pandas as pd

def integrate_and_update(news_df, new_articles, tfidf, tfidf_matrix, user_profiles):
    new_articles_df = pd.DataFrame(new_articles)
    updated_news_df = add_new_articles(news_df, new_articles_df)
    new_tfidf_matrix = update_tfidf_matrix(tfidf, updated_news_df)
    full_tfidf_matrix = sparse.vstack([tfidf_matrix, new_tfidf_matrix])
    
    # Assuming recommend_new_articles function is available and correctly implemented
    recommendations = recommend_new_articles(tfidf, full_tfidf_matrix, user_profiles, new_articles_df)
    return recommendations

def integrate_new_articles_and_recommend(user_profiles, tfidf, tfidf_matrix, news_df, new_articles):
    # Convert new articles to DataFrame
    new_articles_df = pd.DataFrame(new_articles)
    updated_news_df = add_new_articles(news_df, new_articles_df)
    new_tfidf_matrix = update_tfidf_matrix(tfidf, updated_news_df)
    full_tfidf_matrix = sparse.vstack([tfidf_matrix, new_tfidf_matrix])
    
    # Generate new recommendations
    recommendations = recommend_new_articles(tfidf, full_tfidf_matrix, user_profiles, new_articles_df)
    return recommendations



# Load data, build profiles, and save them
dataset_folder = '/data/'
dataset_folder_train = '/data/MINDsmall_train/'
news_file = os.path.join(dataset_folder_train, 'news.tsv')
behaviors_file = os.path.join(dataset_folder_train, 'behaviors.tsv')

# Ensure the dataset folder exists
os.makedirs(dataset_folder_train, exist_ok=True)

dataset_name = 'mind-news-dataset'
kaggle_dataset = 'arashnic/' + dataset_name
#zip_file_path = os.path.join(dataset_folder_train, dataset_name + '.zip')

def extract_zip(file_path, extract_to):
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Extracted {file_path} to {extract_to}")
# Check if files exist, if not download them
if not os.path.exists(news_file) or not os.path.exists(behaviors_file):
    print(f"Dataset files not found. Downloading...")
    subprocess.run(['python', 'download_from_kaggle.py', '--dataset', kaggle_dataset, '--destination', dataset_folder], check=True)
    downloaded_files = os.listdir(dataset_folder_train)
    print(f"Contents of the dataset folder: {downloaded_files}")
#    extract_zip(zip_file_path, dataset_folder_train)
"""
# URL to download the dataset zip
dataset_url = 'https://mind201910small.blob.core.windows.net/release/MINDsmall_train.zip'

def download_file(url, destination):
    response = requests.get(url)
    response.raise_for_status()  # Ensure the request was successful
    with open(destination, 'wb') as f:
        f.write(response.content)
    print(f"Downloaded {url} to {destination}")
"""

os.makedirs(dataset_folder_train, exist_ok=True)

# Check if files exist, if not download and extract them
#if not os.path.exists(news_file) or not os.path.exists(behaviors_file):
#    print(f"Dataset files not found. Downloading...")
#    download_file(dataset_url, zip_file_path)
#    extract_zip(zip_file_path, dataset_folder_train)

# Example Flask route usage
try:
    news_df, behaviors_df = load_and_preprocess_data(f"{dataset_folder_train}news.tsv", f"{dataset_folder_train}behaviors.tsv")
    from models import Model
    from transformers import BertConfig
    from transformers.models.bert.modeling_bert import BertSelfOutput, BertIntermediate, BertOutput
    config=BertConfig.from_json_file('fastformer.json')
    model = Model(config)
    model.load_state_dict(torch.load('/app/downloads/fastformer_model.pth', map_location=torch.device('cpu')))
    model.eval()
    build_user_profiles_and_save_matrix(news_df, behaviors_df, model)
    build_user_profiles_and_save_matrix(news_df, behaviors_df)
    logging.info("user_profiles.pkl created successfully.")
except Exception as e:
    logging.error("Failed to create user_profiles.pkl:", exc_info=True)
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

