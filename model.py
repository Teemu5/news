import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

def load_and_preprocess_data(news_path, behaviors_path):
    news_df = pd.read_csv(news_path, delimiter='\t', names=['NewsID', 'Category', 'SubCategory', 'Title', 'Abstract', 'URL', 'TitleEntities', 'AbstractEntities'])
    behaviors_df = pd.read_csv(behaviors_path, delimiter='\t', names=['ImpressionID', 'UserID', 'Time', 'History', 'Impressions'])
    
    # Preprocess and combine title and abstract for TF-IDF
    news_df['combined_text'] = news_df['Title'].fillna('') + " " + news_df['Abstract'].fillna('')
    return news_df, behaviors_df
def save_data_frames(news_df, behaviors_df, user_profiles, tfidf_matrix):
    with open('news_df.pkl', 'wb') as f:
        pickle.dump(news_df, f)
    with open('user_profiles.pkl', 'wb') as f:
        pickle.dump(user_profiles, f)
    with open('tfidf_matrix.pkl', 'wb') as f:
        pickle.dump(tfidf_matrix, f)

def build_user_profiles_and_save_matrix(news_df, behaviors_df):
    tfidf = TfidfVectorizer(stop_words='english', max_features=1000)
    tfidf_matrix = tfidf.fit_transform(news_df['combined_text'])

    user_profiles = {}
    news_id_index = pd.Series(data=news_df.index, index=news_df.NewsID).to_dict()
    for _, row in behaviors_df.iterrows():
        user_id = row['UserID']
        clicked_articles = [impression.split("-")[0] for impression in row['Impressions'].split() if impression.endswith("-1")]
        indices = [news_id_index[article_id] for article_id in clicked_articles if article_id in news_id_index]
        if indices:
            user_profiles[user_id] = tfidf_matrix[indices].mean(axis=0)

    # Save user profiles and the TF-IDF matrix
    save_data_frames(news_df, behaviors_df, user_profiles, tfidf_matrix)


# Load data, build profiles, and save them
dataset_folder_train = '/mnt/c/Users/OWNER/Downloads/MINDsmall_train/'
# Example Flask route usage

news_df, behaviors_df = load_and_preprocess_data(f"{dataset_folder_train}news.tsv", f"{dataset_folder_train}behaviors.tsv")
build_user_profiles_and_save_matrix(news_df, behaviors_df)
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

