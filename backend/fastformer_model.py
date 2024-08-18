import torch
from transformers import BertConfig

from models import Model
# Load Fastformer model
config = BertConfig.from_json_file('fastformer.json')  # Adjust the path to your config
model = Model(config, word_dict)
model.load_state_dict(torch.load('/path/to/fastformer_model.pth'))  # Adjust path
model.eval()
def tokenize_and_pad(text, word_dict, max_length=256):
    tokens = [word_dict.get(word.lower(), 0) for word in text.split()]  # Simple tokenization
    tokens = tokens[:max_length]  # Truncate to max length
    tokens += [0] * (max_length - len(tokens))  # Pad sequence
    return tokens
def generate_article_embedding(article, model):
    # Tokenize and pad the article text
    tokenized_article = tokenize_and_pad(article['combined_text'], word_dict)
    input_tensor = torch.LongTensor([tokenized_article]).to('cpu')
    
    with torch.no_grad():
        embedding = model(input_tensor)
    
    return embedding.squeeze(0)  # Remove batch dimension

def create_user_profile(articles, model):
    embeddings = []
    for article in articles:
        embedding = generate_article_embedding(article, model)
        embeddings.append(embedding)
    
    # Average the embeddings to create a user profile
    user_profile = torch.mean(torch.stack(embeddings), dim=0)
    return user_profile
