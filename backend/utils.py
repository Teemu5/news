# --- [Imports and Constants] ---
import os
import re
import nltk
import pandas as pd
import numpy as np
import zipfile
from tqdm import tqdm
from nltk.corpus import stopwords
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import Sequence
from keras.models import Model, model_from_json
from tensorflow.keras.layers import Input, Dot, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
import pickle
import tensorflow as tf
from huggingface_hub import hf_hub_download

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# Download NLTK stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def is_colab():
    return 'COLAB_GPU' in os.environ
def init(process_dfs = False, process_behaviors = False):
    global data_dir, vocab_size, max_history_length, max_title_length, news_df, train_df, behaviors_df, user_category_profiles, clustered_data, tokenizer, num_clusters
    zip_file = f"MINDlarge_train.zip"
    valid_zip_file = f"MINDlarge_dev.zip"
    data_dir = 'dataset/train/'  # Adjust path as necessary
    valid_data_dir = 'dataset/valid/'  # Adjust path as necessary
    if is_colab():
        print("Running on Google colab")
        data_dir = '/content/train/'
        valid_data_dir = '/content/valid/'
    #data_dir = 'dataset/small/train/'  # Adjust path as necessary
    #zip_file = f"MINDsmall_train.zip"
    zip_file_path = f"{data_dir}{zip_file}"
    valid_zip_file_path = f"{valid_data_dir}{valid_zip_file}"
    local_model_path = hf_hub_download(
        repo_id=f"Teemu5/news",
        filename=zip_file,
        local_dir=data_dir
    )
    local_model_path = hf_hub_download(
        repo_id=f"Teemu5/news",
        filename=valid_zip_file,
        local_dir=valid_data_dir
    )
    output_folder = os.path.dirname(zip_file_path)
    
    # Unzip the file
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(output_folder)
    if is_colab():
      valid_output_folder = os.path.dirname(valid_zip_file_path)
      with zipfile.ZipFile(valid_zip_file_path, 'r') as zip_ref:
          zip_ref.extractall(os.path.dirname(valid_output_folder))
    news_file = 'news.tsv'
    behaviors_file = 'behaviors.tsv'
    
    # Load news data
    news_path = os.path.join(data_dir, news_file)
    news_df = pd.read_csv(
        news_path,
        sep='\t',
        names=['NewsID', 'Category', 'SubCategory', 'Title', 'Abstract', 'URL', 'TitleEntities', 'AbstractEntities'],
        index_col=False
    )
    
    print("Loaded news data:")
    print(news_df.head())
    
    # Load behaviors data
    behaviors_path = os.path.join(data_dir, behaviors_file)
    behaviors_df = pd.read_csv(
        behaviors_path,
        sep='\t',
        names=['ImpressionID', 'UserID', 'Time', 'HistoryText', 'Impressions'],
        index_col=False
    )
    
    print("\nLoaded behaviors data:")
    print(behaviors_df.head())
    if process_behaviors:
        # Handle missing 'HistoryText' by replacing NaN with empty string
        behaviors_df['HistoryText'] = behaviors_df['HistoryText'].fillna('')
        
        # Create a NewsID to Category mapping
        newsid_to_category = news_df.set_index('NewsID')['Category'].to_dict()
        
        # Function to extract categories from HistoryText
        def extract_categories(history_text):
            if not history_text:
                return []
            news_ids = history_text.split(' ')
            categories = [newsid_to_category.get(news_id, 'Unknown') for news_id in news_ids]
            return categories
        
        # Apply the function to extract categories
        behaviors_df['HistoryCategories'] = behaviors_df['HistoryText'].apply(extract_categories)
        
        print("\nSample HistoryCategories:")
        print(behaviors_df[['UserID', 'HistoryCategories']].head())
        from collections import defaultdict
        
        # Initialize a dictionary to hold category counts per user
        user_category_counts = defaultdict(lambda: defaultdict(int))
        
        # Populate the dictionary
        for idx, row in behaviors_df.iterrows():
            user_id = row['UserID']
            categories = row['HistoryCategories']
            for category in categories:
                user_category_counts[user_id][category] += 1
        
        # Convert the dictionary to a DataFrame
        user_category_profiles = pd.DataFrame(user_category_counts).T.fillna(0)
        
        # Optionally, rename columns to indicate category
        user_category_profiles.columns = [f'Category_{cat}' for cat in user_category_profiles.columns]
        
        print("\nCreated user_category_profiles:")
        print(user_category_profiles.head())
        print(f"\nShape of user_category_profiles: {user_category_profiles.shape}")
        # Handle missing 'HistoryText' by replacing NaN with empty string
        behaviors_df['HistoryText'] = behaviors_df['HistoryText'].fillna('')
        
        # Create a NewsID to Category mapping
        newsid_to_category = news_df.set_index('NewsID')['Category'].to_dict()
        
        # Get all unique UserIDs from behaviors_df
        unique_user_ids = behaviors_df['UserID'].unique()
        
        # Function to extract categories from HistoryText
        def extract_categories(history_text):
            if not history_text:
                return []
            news_ids = history_text.split(' ')
            categories = [newsid_to_category.get(news_id, 'Unknown') for news_id in news_ids]
            return categories
        
        # Apply the function to extract categories
        behaviors_df['HistoryCategories'] = behaviors_df['HistoryText'].apply(extract_categories)
        
        # Explode 'HistoryCategories' to have one category per row
        behaviors_exploded = behaviors_df[['UserID', 'HistoryCategories']].explode('HistoryCategories')
        
        # Replace missing categories with 'Unknown'
        behaviors_exploded['HistoryCategories'] = behaviors_exploded['HistoryCategories'].fillna('Unknown')
        
        # Create a cross-tabulation (pivot table) of counts
        user_category_counts = pd.crosstab(
            index=behaviors_exploded['UserID'],
            columns=behaviors_exploded['HistoryCategories']
        )
        
        # Rename columns to include 'Category_' prefix
        user_category_counts.columns = [f'Category_{col}' for col in user_category_counts.columns]
        
        # Reindex to include all users, filling missing values with zero
        user_category_profiles = user_category_counts.reindex(unique_user_ids, fill_value=0)
        
        print(f"\nCreated user_category_profiles with {user_category_profiles.shape[0]} users and {user_category_profiles.shape[1]} categories.")
        
        # Determine top N categories
        top_n = 20
        category_counts = news_df['Category'].value_counts()
        top_categories = category_counts.nlargest(top_n).index.tolist()
        
        # Get the category names without the 'Category_' prefix
        user_category_columns = user_category_profiles.columns.str.replace('Category_', '')
        
        # Filter columns in user_category_profiles that are in top_categories
        filtered_columns = user_category_profiles.columns[user_category_columns.isin(top_categories)]
        
        # Create filtered_user_category_profiles with these columns
        filtered_user_category_profiles = user_category_profiles[filtered_columns]
        
        # Identify columns that are not in top_categories to sum them into 'Category_Other'
        other_columns = user_category_profiles.columns[~user_category_columns.isin(top_categories)]
        
        # Sum the 'Other' categories
        filtered_user_category_profiles['Category_Other'] = user_category_profiles[other_columns].sum(axis=1)
        
        # Now, get the actual categories present after filtering
        actual_categories = filtered_columns.str.replace('Category_', '').tolist()
        
        # Add 'Other' to the list
        actual_categories.append('Other')
        print(f"Number of new column names: {len(actual_categories)}")
        # Assign new column names
        filtered_user_category_profiles.columns = [f'Category_{cat}' for cat in actual_categories]
        print("\nFiltered user_category_profiles with Top N Categories and 'Other':")
        print(filtered_user_category_profiles.head())
        print(f"\nShape of filtered_user_category_profiles: {filtered_user_category_profiles.shape}")
        
        # Save the user_category_profiles to a file for future use
        user_category_profiles_path = 'user_category_profiles.pkl'
        filtered_user_category_profiles.to_pickle(user_category_profiles_path)
        user_category_profiles = filtered_user_category_profiles
        print(f"\nSaved user_category_profiles to {user_category_profiles_path}")
        behaviors_df.to_pickle("behaviors_df_processed.pkl")
        print(f"\nSaved behaviors_df to behaviors_df_processed.pkl")
    else:
        local_model_path = hf_hub_download(
            repo_id=f"Teemu5/news",
            filename="user_category_profiles.pkl",
            local_dir="models"
        )
        local_model_path = hf_hub_download(
            repo_id=f"Teemu5/news",
            filename="behaviors_df_processed.pkl",
            local_dir="models"
        )
        user_category_profiles = pd.read_pickle("models/user_category_profiles.pkl")
        behaviors_df = pd.read_pickle("models/behaviors_df_processed.pkl")
    print(f"Number of columns in user_category_profiles: {len(user_category_profiles.columns)}")
    # Number of unique users in behaviors_df
    unique_user_ids = behaviors_df['UserID'].unique()
    print(f"Number of unique users in behaviors_df: {len(unique_user_ids)}")
    # Number of unique users in behaviors_df
    unique_user_ids = behaviors_df['UserID'].unique()
    print(f"Number of unique users in behaviors_df: {len(unique_user_ids)}")
    
    # Number of users in user_category_profiles
    user_profile_ids = user_category_profiles.index.unique()
    print(f"Number of users in user_category_profiles: {len(user_profile_ids)}")
    
    # Find missing UserIDs
    missing_user_ids = set(unique_user_ids) - set(user_profile_ids)
    print(f"Number of missing UserIDs in user_category_profiles: {len(missing_user_ids)}")
    tokenizer = Tokenizer()
    num_clusters = 3
    if process_dfs:
        clustered_data, tokenizer, vocab_size, max_history_length, max_title_length, num_clusters = prepare_train_df(
            data_dir=data_dir,
            news_file=news_file,
            behaviours_file=behaviors_file,
            user_category_profiles=user_category_profiles,
            num_clusters=num_clusters,
            fraction=1,
            max_title_length=30,
            max_history_length=50
        )
    local_model_path = hf_hub_download(
        repo_id=f"Teemu5/news",
        filename="news_df_processed.pkl",
        local_dir="models"
    )
    local_model_path = hf_hub_download(
        repo_id=f"Teemu5/news",
        filename="train_df_processed.pkl",
        local_dir="models"
    )
    news_df = pd.read_pickle("models/news_df_processed.pkl")
    train_df = pd.read_pickle("models/train_df_processed.pkl")

    clustered_data = make_clustered_data(train_df, num_clusters)

    tokenizer.fit_on_texts(news_df['CombinedText'].tolist())
    vocab_size = len(tokenizer.word_index) + 1
    max_history_length = 50
    max_title_length = 30
    batch_size = 64 # Adjust as needed
    return data_dir, vocab_size, max_history_length, max_title_length, news_df, train_df, behaviors_df, user_category_profiles, clustered_data, tokenizer, num_clusters
#init()

# --- [Imports and Constants] ---
import os
import re
import nltk
import pandas as pd
import numpy as np
from tqdm import tqdm
from nltk.corpus import stopwords
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import Sequence
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dot, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
import pickle
import tensorflow as tf
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# Download NLTK stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# --- [Cleaning Function] ---
def clean_text(text):
    if pd.isna(text):
        return ''
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Convert to lowercase
    text = text.lower()
    # Remove stopwords
    words = text.split()
    words = [w for w in words if not w in stop_words]
    return ' '.join(words)

def make_clustered_data(train_df, num_clusters):
    clustered_data = {}
    for cluster in range(num_clusters):
        cluster_data = train_df[train_df['Cluster'] == cluster]

        if cluster_data.empty:
            print(f"No data for Cluster {cluster}. Skipping...")
            continue  # Skip to the next cluster

        train_data, val_data = train_test_split(cluster_data, test_size=0.2, random_state=42)
        clustered_data[cluster] = {
            'train': train_data.reset_index(drop=True),
            'val': val_data.reset_index(drop=True)
        }
        print(f"Cluster {cluster}: {len(train_data)} training samples, {len(val_data)} validation samples.")
    return clustered_data

# --- [Data Preparation Function] ---
def prepare_train_df(
    data_dir,
    news_file,
    behaviours_file,
    user_category_profiles,
    num_clusters=3,
    fraction=1,
    max_title_length=30,
    max_history_length=50
):
    # Load news data
    news_path = os.path.join(data_dir, news_file)
    news_df = pd.read_csv(
        news_path,
        sep='\t',
        names=['NewsID', 'Category', 'SubCategory', 'Title', 'Abstract', 'URL', 'TitleEntities', 'AbstractEntities'],
        index_col=False
    )
    print("Loaded news data:")
    print(news_df.head())

    # Load behaviors data
    behaviors_path = os.path.join(data_dir, behaviours_file)
    behaviors_df = pd.read_csv(
        behaviors_path,
        sep='\t',
        names=['ImpressionID', 'UserID', 'Time', 'HistoryText', 'Impressions'],
        index_col=False
    )
    print("Loaded behaviors data:")
    print(behaviors_df.head())

    # Clean titles and abstracts
    news_df['CleanTitle'] = news_df['Title'].apply(clean_text)
    news_df['CleanAbstract'] = news_df['Abstract'].apply(clean_text)

    # Create a combined text field
    news_df['CombinedText'] = news_df['CleanTitle'] + ' ' + news_df['CleanAbstract']

    # Initialize and fit tokenizer
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(news_df['CombinedText'].tolist())
    vocab_size = len(tokenizer.word_index) + 1
    print(f"Vocabulary Size: {vocab_size}")

    # Save tokenizer for future use
    with open('tokenizer.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)

    # Encode and pad CombinedText
    news_df['EncodedText'] = tokenizer.texts_to_sequences(news_df['CombinedText'])
    news_df['PaddedText'] = list(pad_sequences(news_df['EncodedText'], maxlen=max_title_length, padding='post', truncating='post'))

    # Create a mapping from NewsID to PaddedText
    news_text_dict = dict(zip(news_df['NewsID'], news_df['PaddedText']))

    # Function to parse impressions and labels
    def parse_impressions(impressions):
        impression_list = impressions.split()
        news_ids = []
        labels = []
        for imp in impression_list:
            try:
                news_id, label = imp.split('-')
                news_ids.append(news_id)
                labels.append(int(label))
            except ValueError:
                # Handle cases where split does not result in two items
                continue
        return news_ids, labels

    # Apply parsing to behaviors data
    behaviors_df[['ImpressionNewsIDs', 'ImpressionLabels']] = behaviors_df['Impressions'].apply(
        lambda x: pd.Series(parse_impressions(x))
    )

    # Initialize list for train samples
    train_samples = []

    # Iterate over behaviors to create train samples
    for _, row in tqdm(behaviors_df.iterrows(), total=behaviors_df.shape[0]):
        user_id = row['UserID']
        user_cluster = row['Cluster'] if 'Cluster' in row else None  # Cluster will be assigned later

        # Parse user history
        history_ids = row['HistoryText'].split() if pd.notna(row['HistoryText']) else []
        history_texts = [news_text_dict.get(nid, [0]*max_title_length) for nid in history_ids]

        # Limit history length
        if len(history_texts) < max_history_length:
            padding = [[0]*max_title_length] * (max_history_length - len(history_texts))
            history_texts = padding + history_texts
        else:
            history_texts = history_texts[-max_history_length:]

        candidate_news_ids = row['ImpressionNewsIDs']
        labels = row['ImpressionLabels']

        for candidate_news_id, label in zip(candidate_news_ids, labels):
            candidate_text = news_text_dict.get(candidate_news_id, [0]*max_title_length)
            train_samples.append({
                'UserID': user_id,
                'HistoryTitles': history_texts,  # Renamed to 'HistoryTitles'
                'CandidateTitleTokens': candidate_text,  # Renamed to match DataGenerator
                'Label': label
            })

    # Create DataFrame from samples
    train_df = pd.DataFrame(train_samples)
    print(f"Created train_df with {len(train_df)} samples.")
    print("Columns in train_df:")
    print(train_df.columns)
    # --- [Clustering Users] ---
    # Ensure 'UserID's match between user_category_profiles and behaviors_df
    unique_user_ids = behaviors_df['UserID'].unique()
    user_category_profiles = user_category_profiles.loc[unique_user_ids]

    # Check for any missing 'UserID's
    missing_user_ids = set(unique_user_ids) - set(user_category_profiles.index)
    if missing_user_ids:
        print(f"Warning: {len(missing_user_ids)} 'UserID's are missing from user_category_profiles.")
        # Optionally handle missing users
        # For this example, we'll remove these users from behaviors_df
        behaviors_df = behaviors_df[~behaviors_df['UserID'].isin(missing_user_ids)]
    else:
        print("All 'UserID's are present in user_category_profiles.")

    # Perform clustering on user_category_profiles
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    user_clusters = kmeans.fit_predict(user_category_profiles)
    print(f"Assigned clusters to users. Number of clusters: {num_clusters}")

    # Create a DataFrame for user clusters
    user_cluster_df = pd.DataFrame({
        'UserID': user_category_profiles.index,
        'Cluster': user_clusters
    })

    # --- [Assign Clusters Using Map] ---
    print("Assigning cluster labels to train_df using map...")
    user_cluster_mapping = dict(zip(user_cluster_df['UserID'], user_cluster_df['Cluster']))
    train_df['Cluster'] = train_df['UserID'].map(user_cluster_mapping)

    # Verify cluster assignment
    missing_clusters = train_df[train_df['Cluster'].isna()]
    if not missing_clusters.empty:
        print(f"Warning: {len(missing_clusters)} samples have missing cluster assignments.")
        # Remove samples with missing cluster assignments
        train_df = train_df.dropna(subset=['Cluster'])
    else:
        print("All samples have cluster assignments.")

    # Convert 'Cluster' column to integer type
    train_df['Cluster'] = train_df['Cluster'].astype(int)
    for cluster in range(num_clusters):
        cluster_data = train_df[train_df['Cluster'] == cluster]
        print(f"Cluster {cluster}: {len(cluster_data)} test samples.")
    # Assuming train_df has a 'Cluster' column indicating cluster assignments
    # Find the minimum size among all clusters
    min_cluster_size = train_df['Cluster'].value_counts().min()

    # Initialize an empty list to hold balanced data
    balanced_data = []

    # Iterate over each cluster and sample data to balance
    for cluster in train_df['Cluster'].unique():
        cluster_data = train_df[train_df['Cluster'] == cluster]
        balanced_cluster_data = cluster_data.sample(n=min_cluster_size, random_state=42)
        balanced_data.append(balanced_cluster_data)

    # Combine balanced data for all clusters
    balanced_train_df = pd.concat(balanced_data)

    # Update train_df with the balanced data
    # --- [Label Balancing for 0/1 Classes] ---
    # Count how many 0s and 1s we have
    label_counts = balanced_train_df['Label'].value_counts()
    min_label_count = label_counts.min()

    balanced_labels = []
    for label_value in balanced_train_df['Label'].unique():
        label_data = balanced_train_df[balanced_train_df['Label'] == label_value]
        # Downsample to the min_label_count to balance the label distribution
        balanced_label_data = label_data.sample(n=min_label_count, random_state=42)
        balanced_labels.append(balanced_label_data)

    # Combine the label balanced data
    final_balanced_train_df = pd.concat(balanced_labels, ignore_index=True)

    # Shuffle the final dataset to mix up the rows
    final_balanced_train_df = final_balanced_train_df.sample(frac=1, random_state=42).reset_index(drop=True)

    print("\nAfter label balancing (0 vs 1):")
    print(final_balanced_train_df['Label'].value_counts())

    # Now final_balanced_train_df is balanced both by cluster and by label
    train_df = final_balanced_train_df
    #train_df = balanced_train_df.reset_index(drop=True)

    # Print summary of the balanced dataset
    print("Balanced cluster sizes:")
    for cluster in range(num_clusters):
        cluster_data = train_df[train_df['Cluster'] == cluster]
        print(f"Cluster {cluster}: {len(cluster_data)} samples")
    print("Balanced dataset:")
    print(train_df['Cluster'].value_counts())
    """
    clustered_data_balanced = {}
    min_cluster_size = float('inf')
    for cluster in range(num_clusters):
        cluster_data = train_df[train_df['Cluster'] == cluster]
        print(f"Cluster {cluster}: {len(cluster_data)} test samples.")
        min_cluster_size = len(cluster_data) if len(cluster_data) < min_cluster_size else min_cluster_size

    for cluster in range(num_clusters):
        data = train_df[train_df['Cluster'] == cluster]
        if len(data) > min_cluster_size:
            clustered_data_balanced[cluster] = data.sample(n=min_cluster_size, random_state=42)
        else:
            clustered_data_balanced[cluster] = data

    print("Balanced cluster sizes:")
    for cluster, data in clustered_data_balanced.items():
        print(f"Cluster {cluster}: {len(data)} samples")
    """
    # --- [Sampling] ---
    # Optionally perform random sampling
    print(f"Original size: {len(train_df)}")
    train_df_sampled = train_df.sample(frac=fraction, random_state=42)
    print(f"Sampled size: {len(train_df_sampled)}")

    # Optionally, set train_df to sampled
    train_df = train_df_sampled
    print("Columns in sampled train_df:")
    print(train_df.columns)
    print(f"Cluster:{train_df['Cluster']}")
    # --- [Split Data for Each Cluster] ---
    print("Splitting data into training and validation sets for each cluster...")
    clustered_data = {}
    for cluster in range(num_clusters):
        cluster_data = train_df[train_df['Cluster'] == cluster]

        if cluster_data.empty:
            print(f"No data for Cluster {cluster}. Skipping...")
            continue  # Skip to the next cluster

        train_data, val_data = train_test_split(cluster_data, test_size=0.2, random_state=42)
        clustered_data[cluster] = {
            'train': train_data.reset_index(drop=True),
            'val': val_data.reset_index(drop=True)
        }
        print(f"Cluster {cluster}: {len(train_data)} training samples, {len(val_data)} validation samples.")
    print("Saved after processing: models/news_df_processed.pkl, models/train_df_processed.pkl")
    news_df.to_pickle("models/news_df_processed.pkl")
    train_df.to_pickle("models/train_df_processed.pkl")

    return clustered_data, tokenizer, vocab_size, max_history_length, max_title_length, num_clusters

# --- [DataGenerator Class] ---
class DataGenerator(Sequence):
    def __init__(self, df, batch_size, max_history_length=50, max_title_length=30):
        self.df = df.reset_index(drop=True)
        self.batch_size = batch_size
        self.max_history_length = max_history_length
        self.max_title_length = max_title_length
        self.indices = np.arange(len(self.df))
        #print(f"[DataGenerator] Initialized with {len(self.df)} samples and batch_size={self.batch_size}")

    def __len__(self):
        length = int(np.ceil(len(self.df) / self.batch_size))
        #print(f"[DataGenerator] Number of batches per epoch: {length}")
        return length

    def __getitem__(self, idx):
        start = idx * self.batch_size
        end = min((idx + 1) * self.batch_size, len(self.df))
        batch_indices = self.indices[start:end]
        batch_df = self.df.iloc[batch_indices]

        # Debugging: Print batch information
        #print(f"[DataGenerator] Generating batch {idx+1}/{self.__len__()} with samples {start} to {end}")

        if len(batch_df) == 0:
            print(f"[DataGenerator] Warning: Batch {idx} is empty.")
            return None, None

        # Initialize batches
        history_batch = []
        candidate_batch = []
        labels_batch = []

        for _, row in batch_df.iterrows():
            # Get tokenized history titles
            history_titles = row['HistoryTitles']  # List of lists of integers

            # Pad each title in history
            history_titles_padded = pad_sequences(
                history_titles,
                maxlen=self.max_title_length,
                padding='post',
                truncating='post',
                value=0
            )

            # Pad or truncate the history to MAX_HISTORY_LENGTH
            if len(history_titles_padded) < self.max_history_length:
                padding = np.zeros((self.max_history_length - len(history_titles_padded), self.max_title_length), dtype='int32')
                history_titles_padded = np.vstack([padding, history_titles_padded])
            else:
                history_titles_padded = history_titles_padded[-self.max_history_length:]

            # Get candidate title tokens
            candidate_title = row['CandidateTitleTokens']  # List of integers
            candidate_title_padded = pad_sequences(
                [candidate_title],
                maxlen=self.max_title_length,
                padding='post',
                truncating='post',
                value=0
            )[0]

            # Append to batches
            history_batch.append(history_titles_padded)
            candidate_batch.append(candidate_title_padded)
            labels_batch.append(row['Label'])

        # Convert to numpy arrays
        history_batch = np.array(history_batch, dtype='int32')  # Shape: (batch_size, MAX_HISTORY_LENGTH, MAX_TITLE_LENGTH)
        candidate_batch = np.array(candidate_batch, dtype='int32')  # Shape: (batch_size, MAX_TITLE_LENGTH)
        labels_batch = np.array(labels_batch, dtype='float32')  # Shape: (batch_size,)
        inputs = {
            'history_input': history_batch,
            'candidate_input': candidate_batch
        }

        # Debugging: Print shapes
        #print(f"[DataGenerator] Batch shapes - history_input: {history_batch.shape}, candidate_input: {candidate_batch.shape}, labels: {labels_batch.shape}")
        return inputs, labels_batch

    def on_epoch_end(self):
        np.random.shuffle(self.indices)


# --- [Fastformer Model Classes and Functions] ---
from tensorflow.keras.layers import Layer, Dense, Dropout, Softmax, Multiply, Embedding, TimeDistributed, LayerNormalization
from tensorflow.keras.utils import register_keras_serializable

@register_keras_serializable()
class SqueezeLayer(Layer):
    def __init__(self, axis=-1, **kwargs):
        super(SqueezeLayer, self).__init__(**kwargs)
        self.axis = axis

    def call(self, inputs):
        return tf.squeeze(inputs, axis=self.axis)

    def get_config(self):
        config = super(SqueezeLayer, self).get_config()
        config.update({'axis': self.axis})
        return config

@register_keras_serializable()
class ExpandDimsLayer(Layer):
    def __init__(self, axis=-1, **kwargs):
        super(ExpandDimsLayer, self).__init__(**kwargs)
        self.axis = axis

    def call(self, inputs):
        return tf.expand_dims(inputs, axis=self.axis)

    def get_config(self):
        config = super(ExpandDimsLayer, self).get_config()
        config.update({'axis': self.axis})
        return config

@register_keras_serializable()
class SumPooling(Layer):
    def __init__(self, axis=1, **kwargs):
        super(SumPooling, self).__init__(**kwargs)
        self.axis = axis

    def call(self, inputs):
        return tf.reduce_sum(inputs, axis=self.axis)

    def get_config(self):
        config = super(SumPooling, self).get_config()
        config.update({'axis': self.axis})
        return config

@register_keras_serializable()
class Fastformer(Layer):
    def __init__(self, nb_head, size_per_head, **kwargs):
        super(Fastformer, self).__init__(**kwargs)
        self.nb_head = nb_head
        self.size_per_head = size_per_head
        self.output_dim = nb_head * size_per_head

        self.WQ = None
        self.WK = None
        self.WV = None
        self.WO = None

    def build(self, input_shape):
        self.WQ = Dense(self.output_dim, use_bias=False, name='WQ')
        self.WK = Dense(self.output_dim, use_bias=False, name='WK')
        self.WV = Dense(self.output_dim, use_bias=False, name='WV')
        self.WO = Dense(self.output_dim, use_bias=False, name='WO')
        super(Fastformer, self).build(input_shape)

    def call(self, inputs):
        if len(inputs) == 2:
            Q_seq, K_seq = inputs
            Q_mask = None
            K_mask = None
        elif len(inputs) == 4:
            Q_seq, K_seq, Q_mask, K_mask = inputs

        batch_size = tf.shape(Q_seq)[0]
        seq_len = tf.shape(Q_seq)[1]

        # Linear projections
        Q = self.WQ(Q_seq)  # Shape: (batch_size, seq_len, output_dim)
        K = self.WK(K_seq)  # Shape: (batch_size, seq_len, output_dim)
        V = self.WV(K_seq)  # Shape: (batch_size, seq_len, output_dim)

        # Reshape for multi-head attention
        Q = tf.reshape(Q, (batch_size, seq_len, self.nb_head, self.size_per_head))
        K = tf.reshape(K, (batch_size, seq_len, self.nb_head, self.size_per_head))
        V = tf.reshape(V, (batch_size, seq_len, self.nb_head, self.size_per_head))

        # Compute global query and key
        global_q = tf.reduce_mean(Q, axis=1, keepdims=True)  # (batch_size, 1, nb_head, size_per_head)
        global_k = tf.reduce_mean(K, axis=1, keepdims=True)  # (batch_size, 1, nb_head, size_per_head)

        # Compute attention weights
        weights = global_q * K + global_k * Q  # (batch_size, seq_len, nb_head, size_per_head)
        weights = tf.reduce_sum(weights, axis=-1)  # (batch_size, seq_len, nb_head)
        weights = tf.nn.softmax(weights, axis=1)  # Softmax over seq_len

        # Apply attention weights to values
        weights = tf.expand_dims(weights, axis=-1)  # (batch_size, seq_len, nb_head, 1)
        context = weights * V  # (batch_size, seq_len, nb_head, size_per_head)

        # Combine heads
        context = tf.reshape(context, (batch_size, seq_len, self.output_dim))

        # Final projection
        output = self.WO(context)  # (batch_size, seq_len, output_dim)

        return output  # Output shape: (batch_size, seq_len, output_dim)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.output_dim)

    def get_config(self):
        config = super(Fastformer, self).get_config()
        config.update({
            'nb_head': self.nb_head,
            'size_per_head': self.size_per_head
        })
        return config

@register_keras_serializable()
class NewsEncoder(Layer):
    def __init__(self, vocab_size, embedding_dim=256, dropout_rate=0.2, nb_head=8, size_per_head=32, **kwargs):
        super(NewsEncoder, self).__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.dropout_rate = dropout_rate
        self.nb_head = nb_head
        self.size_per_head = size_per_head

        # Define sub-layers
        self.embedding_layer = Embedding(
            input_dim=self.vocab_size,
            output_dim=self.embedding_dim,
            name='embedding_layer'
        )
        self.dropout = Dropout(self.dropout_rate)
        self.dense = Dense(1)
        self.softmax = Softmax(axis=1)
        self.squeeze = SqueezeLayer(axis=-1)
        self.expand_dims = ExpandDimsLayer(axis=-1)
        self.sum_pooling = SumPooling(axis=1)

        self.fastformer_layer = Fastformer(nb_head=self.nb_head, size_per_head=self.size_per_head, name='fastformer_layer')

    def build(self, input_shape):
        super(NewsEncoder, self).build(input_shape)

    def call(self, inputs):
        # Create mask
        mask = tf.cast(tf.not_equal(inputs, 0), dtype='float32')  # Shape: (batch_size, seq_len)

        # Embedding
        title_emb = self.embedding_layer(inputs)  # Shape: (batch_size, seq_len, embedding_dim)
        title_emb = self.dropout(title_emb)

        # Fastformer
        hidden_emb = self.fastformer_layer([title_emb, title_emb, mask, mask])  # Shape: (batch_size, seq_len, embedding_dim)
        hidden_emb = self.dropout(hidden_emb)

        # Attention-based Pooling
        attention_scores = self.dense(hidden_emb)  # Shape: (batch_size, seq_len, 1)
        attention_scores = self.squeeze(attention_scores)  # Shape: (batch_size, seq_len)
        attention_weights = self.softmax(attention_scores)  # Shape: (batch_size, seq_len)
        attention_weights = self.expand_dims(attention_weights)  # Shape: (batch_size, seq_len, 1)
        multiplied = Multiply()([hidden_emb, attention_weights])  # Shape: (batch_size, seq_len, embedding_dim)
        news_vector = self.sum_pooling(multiplied)  # Shape: (batch_size, embedding_dim)

        return news_vector  # Shape: (batch_size, embedding_dim)

    def get_config(self):
        config = super(NewsEncoder, self).get_config()
        config.update({
            'vocab_size': self.vocab_size,
            'embedding_dim': self.embedding_dim,
            'dropout_rate': self.dropout_rate,
            'nb_head': self.nb_head,
            'size_per_head': self.size_per_head
        })
        return config

@register_keras_serializable()
class MaskLayer(Layer):
    def __init__(self, **kwargs):
        super(MaskLayer, self).__init__(**kwargs)

    def call(self, inputs):
        # Create mask: cast to float32 any position that is not equal to zero
        mask = tf.cast(tf.not_equal(inputs, 0), dtype='float32')
        return mask

    def get_config(self):
        config = super(MaskLayer, self).get_config()
        return config

@register_keras_serializable()
class UserEncoder(Layer):
    def __init__(self, news_encoder_layer, embedding_dim=256, **kwargs):
        super(UserEncoder, self).__init__(**kwargs)
        self.news_encoder_layer = news_encoder_layer
        self.embedding_dim = embedding_dim
        self.dropout = Dropout(0.2)
        self.layer_norm = LayerNormalization()
        self.fastformer = Fastformer(nb_head=8, size_per_head=32, name='user_fastformer')
        self.dense = Dense(1)
        self.squeeze = SqueezeLayer(axis=-1)
        self.softmax = Softmax(axis=1)
        self.expand_dims = ExpandDimsLayer(axis=-1)
        self.sum_pooling = SumPooling(axis=1)

    def call(self, inputs):
        # inputs: (batch_size, MAX_HISTORY_LENGTH, MAX_TITLE_LENGTH)
        # Encode each news article in the history
        news_vectors = TimeDistributed(self.news_encoder_layer)(inputs)  # Shape: (batch_size, MAX_HISTORY_LENGTH, embedding_dim)

        # Step 1: Create a boolean mask
        mask = tf.not_equal(inputs, 0)  # Shape: (batch_size, MAX_HISTORY_LENGTH, MAX_TITLE_LENGTH), dtype=bool

        # Step 2: Reduce along the last axis
        mask = tf.reduce_any(mask, axis=-1)  # Shape: (batch_size, MAX_HISTORY_LENGTH), dtype=bool

        # Step 3: Cast to float32 if needed
        mask = tf.cast(mask, dtype='float32')  # Shape: (batch_size, MAX_HISTORY_LENGTH), dtype=float32

        # Fastformer
        hidden_emb = self.fastformer([news_vectors, news_vectors, mask, mask])  # Shape: (batch_size, MAX_HISTORY_LENGTH, embedding_dim)
        hidden_emb = self.dropout(hidden_emb)
        hidden_emb = self.layer_norm(hidden_emb)

        # Attention-based Pooling over history
        attention_scores = self.dense(hidden_emb)  # Shape: (batch_size, MAX_HISTORY_LENGTH, 1)
        attention_scores = self.squeeze(attention_scores)  # Shape: (batch_size, MAX_HISTORY_LENGTH)
        attention_weights = self.softmax(attention_scores)  # Shape: (batch_size, MAX_HISTORY_LENGTH)
        attention_weights = self.expand_dims(attention_weights)  # Shape: (batch_size, MAX_HISTORY_LENGTH, 1)
        multiplied = Multiply()([hidden_emb, attention_weights])  # Shape: (batch_size, MAX_HISTORY_LENGTH, embedding_dim)
        user_vector = self.sum_pooling(multiplied)  # Shape: (batch_size, embedding_dim)

        return user_vector  # Shape: (batch_size, embedding_dim)

    def get_config(self):
        config = super(UserEncoder, self).get_config()
        config.update({
            'embedding_dim': self.embedding_dim,
            'news_encoder_layer': tf.keras.utils.serialize_keras_object(self.news_encoder_layer),
        })
        return config
    @classmethod
    def from_config(cls, config):
        # Extract the serialized news_encoder_layer config
        news_encoder_config = config.pop("news_encoder_layer")
        # Reconstruct the news_encoder_layer instance
        news_encoder_layer = tf.keras.utils.deserialize_keras_object(
            news_encoder_config, custom_objects={'NewsEncoder': NewsEncoder}
        )
        return cls(news_encoder_layer, **config)
# --- [Build Model Function] ---
def build_model(vocab_size, max_title_length=30, max_history_length=50, embedding_dim=256, nb_head=8, size_per_head=32, dropout_rate=0.2):
    # Define Inputs
    history_input = Input(shape=(max_history_length, max_title_length), dtype='int32', name='history_input')
    candidate_input = Input(shape=(max_title_length,), dtype='int32', name='candidate_input')

    # Instantiate NewsEncoder Layer
    news_encoder_layer = NewsEncoder(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        dropout_rate=dropout_rate,
        nb_head=nb_head,
        size_per_head=size_per_head,
        name='news_encoder'
    )

    # Encode Candidate News
    candidate_vector = news_encoder_layer(candidate_input)  # Shape: (batch_size, embedding_dim)

    # Encode User History
    user_vector = UserEncoder(news_encoder_layer, embedding_dim=embedding_dim, name='user_encoder')(history_input)  # Shape: (batch_size, embedding_dim)

    # Scoring Function: Dot Product between User and Candidate Vectors
    score = Dot(axes=-1)([user_vector, candidate_vector])  # Shape: (batch_size, 1)
    score = Activation('sigmoid')(score)  # Shape: (batch_size, 1)

    # Build Model
    model = Model(inputs={'history_input': history_input, 'candidate_input': candidate_input}, outputs=score)
    model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(learning_rate=0.001),
        metrics=[
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='AUC')
        ]
    )

    return model
def build_and_load_weights(weights_file):
    print("""Building model: build_model(
        vocab_size={vocab_size},
        max_title_length={max_title_length},
        max_history_length={max_history_length},
        embedding_dim=256,
        nb_head=8,
        size_per_head=32,
        dropout_rate=0.2
    )""")
    model = build_model(
        vocab_size=vocab_size,
        max_title_length=max_title_length,
        max_history_length=max_history_length,
        embedding_dim=256,
        nb_head=8,
        size_per_head=32,
        dropout_rate=0.2
    )

    # Manually build the model
    input_shapes = {
        'history_input': (None, max_history_length, max_title_length),
        'candidate_input': (None, max_title_length)
    }
    # Prepare dummy inputs
    import numpy as np

    dummy_history_input = np.zeros((1, 50, 30), dtype=np.int32)
    dummy_candidate_input = np.zeros((1, 30), dtype=np.int32)

    # Build the model by passing dummy data
    model.predict({'history_input': dummy_history_input, 'candidate_input': dummy_candidate_input})
    #model.build(input_shapes)
    model.load_weights(weights_file)
    return model
# --- [Training Function] ---
def train_cluster_models(clustered_data, tokenizer, vocab_size, max_history_length, max_title_length, num_clusters, batch_size=64, epochs=5, load_models=[]):
    models = {}

    for cluster in range(num_clusters):
        m_name = f'fastformer_cluster_{cluster}_full_balanced_1_epoch'
        weights_file = f'{m_name}.weights.h5'
        model_file = f'{m_name}.keras'
        model_h5_file = f'{m_name}.h5'
        model_hdf5_file = f'{m_name}.hdf5'
        model_json_file = f'{m_name}.json'
        if cluster in load_models: # load_models should be list of number indicating which models to load and not train
            print(f"\nLoading model for Cluster {cluster} from {model_hdf5_file}")
            local_model_path = hf_hub_download(
                repo_id=f"Teemu5/news",
                filename=model_hdf5_file,
                local_dir="."
            )
            from tensorflow.keras.utils import custom_object_scope
            with custom_object_scope({'UserEncoder': UserEncoder, 'NewsEncoder': NewsEncoder}):
                model = tf.keras.models.load_model(model_hdf5_file)#build_and_load_weights(weights_file)
                models[cluster] = model
            #model.save(model_file)
            #print(f"Saved model for Cluster {cluster} into {model_file}.")
            continue
        print(f"\nTraining model for Cluster {cluster} into {weights_file}")
        # Retrieve training and validation data
        train_data = clustered_data[cluster]['train']
        val_data = clustered_data[cluster]['val']

        print(f"Cluster {cluster} - Training samples: {len(train_data)}, Validation samples: {len(val_data)}")

        # Create data generators
        train_generator = DataGenerator(train_data, batch_size, max_history_length, max_title_length)
        val_generator = DataGenerator(val_data, batch_size, max_history_length, max_title_length)

        steps_per_epoch = len(train_generator)
        validation_steps = len(val_generator)

        # Build model
        model = build_model(
            vocab_size=vocab_size,
            max_title_length=max_title_length,
            max_history_length=max_history_length,
            embedding_dim=256,
            nb_head=8,
            size_per_head=32,
            dropout_rate=0.2
        )
        print(model.summary())

        # Define callbacks
        early_stopping = EarlyStopping(
            monitor='val_AUC',
            patience=2,
            mode='max',
            restore_best_weights=True
        )
        csv_logger = CSVLogger(f'training_log_cluster_{cluster}.csv', append=True)
        model_checkpoint = ModelCheckpoint(
            f'best_model_cluster_{cluster}.keras',
            monitor='val_AUC',
            mode='max',
            save_best_only=True
        )

        # Train the model
        model.fit(
            train_generator,
            epochs=epochs,
            #steps_per_epoch=steps_per_epoch,
            #validation_data=val_generator,
            validation_steps=validation_steps,
            callbacks=[early_stopping, csv_logger, model_checkpoint]
        )

        # Save model weights
        model.save_weights(weights_file)
        print(f"Saved model weights for Cluster {cluster} into {weights_file}.")
        model.save(model_h5_file)
        print(f"Saved h5 model for Cluster {cluster} into {model_h5_file}.")
        model.save(model_file)
        print(f"Saved model for Cluster {cluster} into {model_file}.")

        # Store the model
        models[cluster] = model
        # Clear memory
        del train_data, val_data, train_generator, val_generator, model
        import gc
        gc.collect()
    print("Returning models list")
    print(models)
    return models

# --- [Recommendation Function] ---
def recommend_news(user_id, user_cluster_df, models, candidate_texts, history_texts, max_history_length=50, max_title_length=30):
    # Determine the user's cluster
    cluster = user_cluster_df.get(user_id)
    if cluster is None:
        print(f"User {user_id} not found in any cluster.")
        return None

    # Retrieve the corresponding model
    model = models.get(cluster)
    if model is None:
        print(f"No model trained for Cluster {cluster}.")
        return None

    # Prepare input data for the user
    history_padded = pad_sequences(
        [history_texts],
        maxlen=max_history_length,
        padding='post',
        truncating='post',
        value=0
    )  # Shape: (1, max_history_length, max_title_length)

    candidate_padded = pad_sequences(
        [candidate_texts],
        maxlen=max_title_length,
        padding='post',
        truncating='post',
        value=0
    )  # Shape: (1, max_title_length)

    inputs = {
        'history_input': history_padded,
        'candidate_input': candidate_padded
    }

    # Generate prediction
    prediction = model.predict(inputs)[0][0]  # Get the prediction score

    # Return the prediction score
    return prediction
# --- [Main Execution] ---
def main():
    # Paths to data files
    news_file = 'news.tsv'
    behaviors_file = 'behaviors.tsv'

    # Load behaviors data to get unique UserIDs
    behaviors_path = os.path.join(data_dir, behaviors_file)
    behaviors_df = pd.read_csv(
        behaviors_path,
        sep='\t',
        names=['ImpressionID', 'UserID', 'Time', 'HistoryText', 'Impressions'],
        index_col=False
    )

    # Extract unique UserIDs
    unique_user_ids = behaviors_df['UserID'].unique()
    print(f"Number of unique users in behaviors_df: {len(unique_user_ids)}")

    # Create dummy user_category_profiles with matching UserIDs
    #user_category_profiles = pd.DataFrame(
    #    np.random.rand(len(unique_user_ids), 10),  # One row per user
    #    index=unique_user_ids,
    #    columns=[f'feature_{j}' for j in range(10)]
    #)

    # Prepare clustered data
    clustered_data, tokenizer, vocab_size, max_history_length, max_title_length, num_clusters = prepare_train_df(
        data_dir=data_dir,
        news_file=news_file,
        behaviours_file=behaviors_file,
        user_category_profiles=user_category_profiles,
        num_clusters=3,
        fraction=1,
        max_title_length=30,
        max_history_length=50
    )

    # Train cluster-specific models
    models = train_cluster_models(
        clustered_data=clustered_data,
        tokenizer=tokenizer,
        vocab_size=vocab_size,
        max_history_length=max_history_length,
        max_title_length=max_title_length,
        num_clusters=num_clusters,
        batch_size=64,
        epochs=1
    )

def init_old():
    global vocab_size, max_history_length, max_title_length
    zip_file = f"MINDlarge_train.zip"
    valid_zip_file = f"MINDlarge_test.zip"
    data_dir = 'dataset/train/'  # Adjust path as necessary
    valid_data_dir = 'dataset/valid/'  # Adjust path as necessary
    if is_colab():
        print("Running on Google colab")
        data_dir = '/content/train/'
        valid_data_dir = '/content/valid/'
    #data_dir = 'dataset/small/train/'  # Adjust path as necessary
    #zip_file = f"MINDsmall_train.zip"
    zip_file_path = f"{data_dir}{zip_file}"
    valid_zip_file_path = f"{valid_data_dir}{valid_zip_file}"
    # Get the directory where the zip file is located
    output_folder = os.path.dirname(zip_file_path)
    # Unzip the file
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(output_folder)
    if is_colab():
        valid_output_folder = os.path.dirname(valid_zip_file_path)
        with zipfile.ZipFile(valid_zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(os.path.dirname(valid_output_folder))
    news_file = 'news.tsv'
    behaviors_file = 'behaviors.tsv'

    # Load news data
    news_path = os.path.join(data_dir, news_file)
    news_df = pd.read_csv(
        news_path,
        sep='\t',
        names=['NewsID', 'Category', 'SubCategory', 'Title', 'Abstract', 'URL', 'TitleEntities', 'AbstractEntities'],
        index_col=False
    )

    print("Loaded news data:")
    print(news_df.head())

    # Load behaviors data
    behaviors_path = os.path.join(data_dir, behaviors_file)
    behaviors_df = pd.read_csv(
        behaviors_path,
        sep='\t',
        names=['ImpressionID', 'UserID', 'Time', 'HistoryText', 'Impressions'],
        index_col=False
    )

    print("\nLoaded behaviors data:")
    print(behaviors_df.head())
    # Handle missing 'HistoryText' by replacing NaN with empty string
    behaviors_df['HistoryText'] = behaviors_df['HistoryText'].fillna('')

    # Create a NewsID to Category mapping
    newsid_to_category = news_df.set_index('NewsID')['Category'].to_dict()

    # Function to extract categories from HistoryText
    def extract_categories(history_text):
        if not history_text:
            return []
        news_ids = history_text.split(' ')
        categories = [newsid_to_category.get(news_id, 'Unknown') for news_id in news_ids]
        return categories

    # Apply the function to extract categories
    behaviors_df['HistoryCategories'] = behaviors_df['HistoryText'].apply(extract_categories)

    print("\nSample HistoryCategories:")
    print(behaviors_df[['UserID', 'HistoryCategories']].head())
    from collections import defaultdict

    # Initialize a dictionary to hold category counts per user
    user_category_counts = defaultdict(lambda: defaultdict(int))

    # Populate the dictionary
    for idx, row in behaviors_df.iterrows():
        user_id = row['UserID']
        categories = row['HistoryCategories']
        for category in categories:
            user_category_counts[user_id][category] += 1

    # Convert the dictionary to a DataFrame
    user_category_profiles = pd.DataFrame(user_category_counts).T.fillna(0)

    # Optionally, rename columns to indicate category
    user_category_profiles.columns = [f'Category_{cat}' for cat in user_category_profiles.columns]

    print("\nCreated user_category_profiles:")
    print(user_category_profiles.head())
    print(f"\nShape of user_category_profiles: {user_category_profiles.shape}")
    # Handle missing 'HistoryText' by replacing NaN with empty string
    behaviors_df['HistoryText'] = behaviors_df['HistoryText'].fillna('')

    # Create a NewsID to Category mapping
    newsid_to_category = news_df.set_index('NewsID')['Category'].to_dict()

    # Get all unique UserIDs from behaviors_df
    unique_user_ids = behaviors_df['UserID'].unique()

    tokenizer = Tokenizer()
    clustered_data, tokenizer, vocab_size, max_history_length, max_title_length, num_clusters = prepare_train_df(
        data_dir=data_dir,
        news_file=news_file,
        behaviours_file=behaviors_file,
        user_category_profiles=user_category_profiles,
        num_clusters=3,
        fraction=1,
        max_title_length=30,
        max_history_length=50
    )

    tokenizer.fit_on_texts(news_df['CombinedText'].tolist())
    vocab_size = len(tokenizer.word_index) + 1
    max_history_length = 50
    max_title_length = 30
    batch_size = 64  # Adjust as needed
def build_and_load_weights(weights_file):
    init()
    model = build_model(
        vocab_size=vocab_size,
        max_title_length=max_title_length,
        max_history_length=max_history_length,
        embedding_dim=256,
        nb_head=8,
        size_per_head=32,
        dropout_rate=0.2
    )

    # Manually build the model
    input_shapes = {
        'history_input': (None, max_history_length, max_title_length),
        'candidate_input': (None, max_title_length)
    }
    # Prepare dummy inputs
    import numpy as np

    dummy_history_input = np.zeros((1, 50, 30), dtype=np.int32)
    dummy_candidate_input = np.zeros((1, 30), dtype=np.int32)

    # Build the model by passing dummy data
    model.predict({'history_input': dummy_history_input, 'candidate_input': dummy_candidate_input})
    #model.build(input_shapes)
    model.load_weights(weights_file)
    return model
def load_model_and_weights(weights_file, model_json_file):
    json_file = open(model_json_file, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    with tf.keras.utils.custom_object_scope({'UserEncoder': UserEncoder,
                                             'NewsEncoder': NewsEncoder}):
        model = model_from_json(loaded_model_json)
        model.load_weights(weights_file)
def get_models(process_dfs = False, process_behaviors = False):
    news_file = 'news.tsv'
    behaviors_file = 'behaviors.tsv'
    data_dir, vocab_size, max_history_length, max_title_length, news_df, train_df, behaviors_df, user_category_profiles, clustered_data, tokenizer, num_clusters = init(process_dfs, process_behaviors)
    if process_dfs:
        clustered_data, tokenizer, vocab_size, max_history_length, max_title_length, num_clusters = prepare_train_df(
            data_dir=data_dir,
            news_file=news_file,
            behaviours_file=behaviors_file,
            user_category_profiles=user_category_profiles,
            num_clusters=3,
            fraction=1,
            max_title_length=30,
            max_history_length=50
        )
    models = train_cluster_models(
        clustered_data=clustered_data,
        tokenizer=tokenizer,
        vocab_size=vocab_size,
        max_history_length=max_history_length,
        max_title_length=max_title_length,
        num_clusters=num_clusters,
        batch_size=64,
        epochs=1,
        load_models=[0,1,2,3]
    )
    return models