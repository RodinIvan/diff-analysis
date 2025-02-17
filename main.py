import glob
import numpy
import sklearn
import lightgbm as lgb

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, classification_report
from collections import defaultdict, Counter
from pathlib import Path
import os


csv_files = glob.glob("/data/*.csv")

if len(csv_files) > 0:
    print("Yay! I can read the data:")
    print(csv_files)
else:
    print("Darn! I cannot read the data at /data.")
    print("Probably because the ./data dir on the host is not mounted correctly to /data in the container.")

print(csv_files)
df = pd.read_csv(csv_files[-1])

df = df[df['old_file'] == df['new_file']]  # Ignore renames

file_change_counts = df['new_file'].value_counts()
filtered_files = file_change_counts[file_change_counts >= 3].index
df_filtered = df[df['new_file'].isin(filtered_files)].copy()
print(f"Filtered dataset has {len(df_filtered)} rows. Total number of files: {len(filtered_files)}")

df = df_filtered

# 2. Group by commits and sort chronologically
commits = df.groupby('child_sha').agg({
    'old_file': lambda x: list(x),
    'when': 'first',
    'new_author': 'first',
    'old_author': 'first',
    'old_lines': 'first',
    'new_lines': 'first',
}).reset_index().sort_values('when')

# 3. Train-test split (70/30 temporal split)
split_idx = int(len(commits) * 0.7)
train_commits = commits.iloc[:split_idx]
test_commits = commits.iloc[split_idx:]

# 4. Feature engineering preparations
# Co-occurrence statistics
co_occur_counts = defaultdict(lambda: defaultdict(int))
file_counts = defaultdict(int)

# Extract all file extensions
extensions = [Path(file).suffix.lower() for file in df['old_file']]

# Count frequency of each extension
extension_counts = Counter(extensions)

# Print the most common extensions
print("Most common file extensions:")
for ext, count in extension_counts.most_common(10):  # Top 10 extensions
    print(f"{ext}: {count} occurrences")

# Select the most common extensions (e.g., top 5 or top 10)
common_extensions = [ext for ext, _ in extension_counts.most_common(5)] 

# Calculate co-occurrence from training data
for _, row in train_commits.iterrows():
    files = row['old_file']
    for i in range(len(files)):
        for j in range(i+1, len(files)):
            a, b = files[i], files[j]
            co_occur_counts[a][b] += 1
            co_occur_counts[b][a] += 1
    for f in files:
        file_counts[f] += 1

# File distance calculation
def get_file_distance(a, b):
    a_parts = Path(a).parts[:-1]
    b_parts = Path(b).parts[:-1]
    
    min_len = min(len(a_parts), len(b_parts))
    common = 0
    for i in range(min_len):
        if a_parts[i] == b_parts[i]:
            common += 1
        else:
            break
    return (len(a_parts) - common) + (len(b_parts) - common)

# 1. Enhanced feature engineering
def extract_features(a, b, author_changed, line_changes_norm, co_occur_counts, file_counts):
    features = [
        # Existing features
        co_occur_counts[a].get(b, 0) / max(file_counts[a], 1),  # Co-occurrence
        author_changed,  # Author changed
        get_file_distance(a, b),  # File distance
        
        # New features
        line_changes_norm[0],  # Normalized line changes
        line_changes_norm[1],  # Normalized line changes
        *file_extension_features(a)  # File extension features
    ]
    return features

def file_extension_features(file_path):
    ext = Path(file_path).suffix.lower()
    common_extensions = ['.c', '.h', '.py', '.java', '.md']  # Add more based on your data
    return [int(ext == e) for e in common_extensions]

# 3. Modified training data creation
X_train = []
y_train = []

for _, row in train_commits.iterrows():
    files = row['old_file']
    author_changed = int(row['new_author'] != row['old_author'])
    # line_changes = (row['old_lines'] + row['new_lines']) / max_lines  # Normalized
    line_changes = (row['old_lines'], row['new_lines'])
    # print(line_changes)
    # Positive samples
    for i in range(len(files)):
        for j in range(i+1, len(files)):
            a, b = files[i], files[j]
            features = extract_features(a, b, author_changed, line_changes, 
                                      co_occur_counts, file_counts)
            X_train.append(features)
            y_train.append(1)
            
            # Symmetric pair
            features = extract_features(b, a, author_changed, line_changes,
                                      co_occur_counts, file_counts)
            X_train.append(features)
            y_train.append(1)
    
    # Negative samples (rest remains similar but using extract_features)
    # Negative samples (random pairs)
    all_files = list(file_counts.keys())
    for a in files:
        # Sample some files not in commit
        # non_files = [f for f in all_files if f not in files]
        non_files = np.random.choice(
            [f for f in all_files if f not in files], 
            size=min(3, len(files)),  # Balance pos/neg
            replace=False
        )
        for b in non_files:
            features = extract_features(a, b, author_changed, line_changes, 
                                      co_occur_counts, file_counts)
            X_train.append(features)
            # co_occur = co_occur_counts[a][b] / max(file_counts[a], 1)
            # distance = get_file_distance(a, b)
            # X_train.append([co_occur, author_changed, distance])
            y_train.append(0)
 
print(f"Finished training data creation with {len(X_train)} samples")  


# 4. Train LightGBM with feature importance tracking
model = lgb.LGBMClassifier(
    n_estimators=100,  # Number of boosting rounds
    learning_rate=0.05,  # Smaller learning rate for better generalization
    max_depth=4,  # Maximum tree depth
    random_state=42,  # For reproducibility
    is_unbalance=True,  # Handle class imbalance
    metric='binary_logloss',  # Metric to optimize
    verbose=-1  # Suppress LightGBM logs
)

model.fit(X_train, y_train)

print('Finished training')

# 6. Threshold-based evaluation
THRESHOLD = 0.5  #Can be tuned using validation set

y_true = []
y_pred_probs = []
all_files = list(file_counts.keys())

precision_scores = []
recall_scores = []

for _, row in test_commits.iterrows():
    files = row['old_file']
    if len(files) < 2:
        continue
    author_changed = int(row['new_author'] != row['old_author'])
    line_changes = (row['old_lines'], row['new_lines'])
    for target_file in files:
        true_partners = [f for f in files if f != target_file]
        if not true_partners:
            continue
            
        # Create all possible pairs (file vs rest of the system)
        candidates = []
        for candidate_file in all_files:
            if candidate_file == target_file:
                continue
                
            features = extract_features(
                target_file, candidate_file,
                author_changed, line_changes,
                co_occur_counts, file_counts
            )
            candidates.append( (candidate_file, features) )
            
        # Predict probabilities
        X_test = [x for (_, x) in candidates]
        if not X_test:
            continue
            
        probs = model.predict_proba(X_test)[:, 1]
        
        # Apply threshold
        predictions = (probs >= THRESHOLD).astype(int)
        
        # Get ground truth labels
        true_labels = [1 if f in true_partners else 0 for f in all_files if f != target_file]
        
        # Store results
        y_true.extend(true_labels)
        y_pred_probs.extend(probs)

        precision_scores.append(precision_score(true_labels, predictions, zero_division=0))
        recall_scores.append(recall_score(true_labels, predictions, zero_division=0)) 

# Convert probabilities to final predictions
y_pred = (np.array(y_pred_probs) >= THRESHOLD).astype(int)

print(f"AVG Precision: {np.mean(precision_scores):.4f}")
print(f"AVG Recall: {np.mean(recall_scores):.4f}")  

# 7. Enhanced evaluation metrics
print("\nClassification Report:")
print(classification_report(y_true, y_pred, zero_division=0))
precision = precision_score(y_true, y_pred, zero_division=0)
recall = recall_score(y_true, y_pred, zero_division=0)
print(f"Precision: {precision_score(y_true, y_pred, zero_division=0):.4f}")
print(f"Recall: {recall_score(y_true, y_pred, zero_division=0):.4f}")
print(f"F1 Score: {2*(precision*recall)/(precision+recall):.4f}" if (precision+recall) > 0 else "F1: Undefined")
