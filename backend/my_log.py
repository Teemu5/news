# IPython log file

# Install required dependencies
#!pip install --upgrade pip
#!pip install numpy scikit-learn tensorflow keras fastapi torch transformers

# If your project has a requirements.txt file, you can also use:
# !pip install --no-cache-dir -r requirements.txt
#[Out]#    user_id              ref_date  num_candidates recommended_ids  \
#[Out]# 0  U185223  2019-11-09T00:00:00Z               0              []   
#[Out]# 
#[Out]#    precision@20  recall@20  inference_time  
#[Out]# 0           0.0          0        2.243522  
thresholds = np.linspace(0.0, 1.0, 21)
performance = []

for thr in thresholds:
    filtered_candidates = [score for score in candidate_similarities if score >= thr]
    # Here, compute your chosen metric (e.g., F1 or Precision@K) using the filtered candidates.
    metric_value = compute_metric(filtered_candidates, ground_truth_clicks)
    performance.append(metric_value)

plt.plot(thresholds, performance, marker='o')
plt.xlabel("TF‑IDF Similarity Threshold")
plt.ylabel("Metric (e.g., F1‑score)")
plt.title("Threshold Tuning for Candidate Filtering")
plt.show()
