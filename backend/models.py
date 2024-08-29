import nltk
nltk.download('punkt')


import torch
from datasets import load_dataset
from nltk.tokenize import wordpunct_tokenize
dataset = load_dataset('ag_news')
text=[]
label=[]
for row in dataset['train']['text']+dataset['test']['text']:
    text.append(wordpunct_tokenize(row.lower()))
for row in dataset['train']['label']+dataset['test']['label']:
    label.append(row)
word_dict={'PADDING':0}
for sent in text:
    for token in sent:
        if token not in word_dict:
            word_dict[token]=len(word_dict)
news_words = []
for sent in text:
    sample=[]
    for token in sent:
        sample.append(word_dict[token])
    sample = sample[:256]
    news_words.append(sample+[0]*(256-len(sample)))
import numpy as np
news_words=np.array(news_words,dtype='int32')
label=np.array(label,dtype='int32')
index=np.arange(len(label))
train_index=index[:120000]
np.random.shuffle(train_index)
test_index=index[120000:]
import os
import logging
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


from transformers import BertConfig
from transformers.models.bert.modeling_bert import BertSelfOutput, BertIntermediate, BertOutput
config=BertConfig.from_json_file('fastformer.json')
import torch
import torch.nn as nn

class AttentionPooling(nn.Module):
    def __init__(self, config):
        self.config = config
        super(AttentionPooling, self).__init__()
        self.att_fc1 = nn.Linear(config.hidden_size, config.hidden_size)
        self.att_fc2 = nn.Linear(config.hidden_size, 1)
        self.apply(self.init_weights)

    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


    def forward(self, x, attn_mask=None):
        bz = x.shape[0]
        e = self.att_fc1(x)
        e = nn.Tanh()(e)
        alpha = self.att_fc2(e)
        alpha = torch.exp(alpha)
        if attn_mask is not None:
            alpha = alpha * attn_mask.unsqueeze(2)
        alpha = alpha / (torch.sum(alpha, dim=1, keepdim=True) + 1e-8)
        x = torch.bmm(x.permute(0, 2, 1), alpha)
        x = torch.reshape(x, (bz, -1))
        return x

class FastSelfAttention(nn.Module):
    def __init__(self, config):
        super(FastSelfAttention, self).__init__()
        self.config = config
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" %
                (config.hidden_size, config.num_attention_heads))
        self.attention_head_size = int(config.hidden_size /config.num_attention_heads)
        self.num_attention_heads = config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.input_dim= config.hidden_size

        self.query = nn.Linear(self.input_dim, self.all_head_size)
        self.query_att = nn.Linear(self.all_head_size, self.num_attention_heads)
        self.key = nn.Linear(self.input_dim, self.all_head_size)
        self.key_att = nn.Linear(self.all_head_size, self.num_attention_heads)
        self.transform = nn.Linear(self.all_head_size, self.all_head_size)

        self.softmax = nn.Softmax(dim=-1)

        self.apply(self.init_weights)

    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads,
                                       self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask):
        # batch_size, seq_len, num_head * head_dim, batch_size, seq_len
        batch_size, seq_len, _ = hidden_states.shape
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        # batch_size, num_head, seq_len
        query_for_score = self.query_att(mixed_query_layer).transpose(1, 2) / self.attention_head_size**0.5
        # add attention mask
        query_for_score += attention_mask

        # batch_size, num_head, 1, seq_len
        query_weight = self.softmax(query_for_score).unsqueeze(2)

        # batch_size, num_head, seq_len, head_dim
        query_layer = self.transpose_for_scores(mixed_query_layer)

        # batch_size, num_head, head_dim, 1
        pooled_query = torch.matmul(query_weight, query_layer).transpose(1, 2).view(-1,1,self.num_attention_heads*self.attention_head_size)
        pooled_query_repeat= pooled_query.repeat(1, seq_len,1)
        # batch_size, num_head, seq_len, head_dim

        # batch_size, num_head, seq_len
        mixed_query_key_layer=mixed_key_layer* pooled_query_repeat

        query_key_score=(self.key_att(mixed_query_key_layer)/ self.attention_head_size**0.5).transpose(1, 2)

        # add attention mask
        query_key_score +=attention_mask

        # batch_size, num_head, 1, seq_len
        query_key_weight = self.softmax(query_key_score).unsqueeze(2)

        key_layer = self.transpose_for_scores(mixed_query_key_layer)
        pooled_key = torch.matmul(query_key_weight, key_layer)

        #query = value
        weighted_value =(pooled_key * query_layer).transpose(1, 2)
        weighted_value = weighted_value.reshape(
            weighted_value.size()[:-2] + (self.num_attention_heads * self.attention_head_size,))
        weighted_value = self.transform(weighted_value) + mixed_query_layer

        return weighted_value
class FastAttention(nn.Module):
    def __init__(self, config):
        super(FastAttention, self).__init__()
        self.self = FastSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor, attention_mask):
        self_output = self.self(input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output

class FastformerLayer(nn.Module):
    def __init__(self, config):
        super(FastformerLayer, self).__init__()
        self.attention = FastAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask):
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output

class FastformerEncoder(nn.Module):
    def __init__(self, config, pooler_count=1):
        super(FastformerEncoder, self).__init__()
        self.config = config
        self.encoders = nn.ModuleList([FastformerLayer(config) for _ in range(config.num_hidden_layers)])
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # support multiple different poolers with shared bert encoder.
        self.poolers = nn.ModuleList()
        if config.pooler_type == 'weightpooler':
            for _ in range(pooler_count):
                self.poolers.append(AttentionPooling(config))
        logging.info(f"This model has {len(self.poolers)} poolers.")

        self.apply(self.init_weights)

    def init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if isinstance(module, (nn.Embedding)) and module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].fill_(0)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self,
                input_embs,
                attention_mask,
                pooler_index=0):
        #input_embs: batch_size, seq_len, emb_dim
        #attention_mask: batch_size, seq_len, emb_dim

        extended_attention_mask = attention_mask.unsqueeze(1)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        batch_size, seq_length, emb_dim = input_embs.shape
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_embs.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        position_embeddings = self.position_embeddings(position_ids)

        embeddings = input_embs + position_embeddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        #print(embeddings.size())
        all_hidden_states = [embeddings]

        for i, layer_module in enumerate(self.encoders):
            layer_outputs = layer_module(all_hidden_states[-1], extended_attention_mask)
            all_hidden_states.append(layer_outputs)
        assert len(self.poolers) > pooler_index
        output = self.poolers[pooler_index](all_hidden_states[-1], attention_mask)

        return output

class Model(torch.nn.Module):

    def __init__(self,config):
        super(Model, self).__init__()
        self.config = config
        self.dense_linear = nn.Linear(config.hidden_size,4)
        self.word_embedding = nn.Embedding(len(word_dict),256,padding_idx=0)
        self.fastformer_model = FastformerEncoder(config)
        self.criterion = nn.CrossEntropyLoss()
        self.apply(self.init_weights)

    def init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if isinstance(module, (nn.Embedding)) and module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].fill_(0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self,input_ids,targets,error_msg=""):
        try:
            print(f"t")
            error_msg += f"t"
            #input_ids = input_ids.long() 
            def preprocess_input_ids(input_ids, valid_range): # Why are there negatives?
                # Replace negative values or out-of-range values with padding index (0)
                input_ids = torch.clamp(input_ids, min=0, max=valid_range-1)
                return input_ids
            valid_range = len(self.word_embedding.weight)  # Assuming valid indices are within [0, valid_range-1]
            input_ids = preprocess_input_ids(input_ids, valid_range)

            mask=input_ids.bool().float()
            print(f"mask:{mask}")
            error_msg += f"mask:{mask}"
            embds=self.word_embedding(input_ids)
            print(f"embds:{embds}")
            error_msg += f"embds:{embds}"
            text_vec = self.fastformer_model(embds,mask)
            print(f"text_vec:{text_vec}")
            error_msg += f"text_vec:{text_vec}"
            score = self.dense_linear(text_vec)
            print(f"targets:{targets}")
            error_msg += f"targets:{targets}"
            loss = self.criterion(score, targets)
            return loss, score

        except Exception as e:
            return f"error: {e} {error_msg}"
import torch
from transformers import BertConfig
config = BertConfig.from_json_file('fastformer.json')
model = Model(config)
#model.load_state_dict(torch.load('/path/to/fastformer_model.pth'))
model.load_state_dict(torch.load('/app/downloads/fastformer_model.pth', map_location=torch.device('cpu')))
model.eval()
def tokenize_and_pad(text, word_dict, max_length=256):
    tokens = [word_dict.get(word.lower(), 0) for word in text.split()]
    tokens = tokens[:max_length]
    tokens += [0] * (max_length - len(tokens))
    return tokens
def generate_article_embedding(article, model):
    # Tokenize and pad the article text
    tokenized_article = tokenize_and_pad(article['combined_text'], word_dict)
    input_tensor = torch.LongTensor([tokenized_article]).to('cpu')
    
    with torch.no_grad():
        embedding = model(input_tensor)
    
    return embedding.squeeze(0)

def create_user_profile(articles, model):
    embeddings = []
    for article in articles:
        embedding = generate_article_embedding(article, model)
        embeddings.append(embedding)

    user_profile = torch.mean(torch.stack(embeddings), dim=0)
    return user_profile
def recommend_new_articles(model, user_profiles, new_articles):
    # Generate embeddings for the new articles
    new_article_embeddings = []
    for article in new_articles:
        embedding = generate_article_embedding(article, model)
        new_article_embeddings.append(embedding)
    
    new_article_embeddings = torch.stack(new_article_embeddings)
    
    # Compute recommendations for each user based on cosine similarity
    recommendations = {}
    for user_id, profile in user_profiles.items():
        sim_scores = torch.nn.functional.cosine_similarity(profile.unsqueeze(0), new_article_embeddings)
        top_indices = torch.topk(sim_scores, k=10).indices.cpu().numpy().tolist()  # Top 10 recommendations
        recommendations[user_id] = [new_articles[i]['NewsID'] for i in top_indices]
    
    return recommendations