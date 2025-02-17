*Data Exploration, task formulation*

First, I didn't fully get what is expected from me in the task. One of my assumption was that from commit data at timtestamp [t] I should predict a list of files will be changed in the following commit at [t+1].

However, this analysis have shown, that in fact, the intersection of files between the consequent commits is usually very small - the average Jaccard similarity between the sets of files in conseuqent commits is just around 0.013. Also the correlation between the number of changed lines and the appearance of file in the next commit is very small, so I decided, that we need to make predictions within the files from the same commit.

To make the problem simpler from evaluation perspective, I reformulate it as following:
If total number of files in the dataset is equal to N, then given the fact that file x_i is changed in the commit, the model has to make (N-1) predictions for all x_j, j!=i.

*Data cleaning*

I worked only with  the 'ffmpeg-master-none.csv'. This file doesn't contain renaming.

As my simplest baseline is built only on co-occurrences analysis, I decided to exclude all the files which appear in the csv less than 3 files. After this filtration, 1208 files are left.

As the data is temporally dependent, I perform train/val split based on the time of commit - first 70% go to train, remaining to validation.

*Simplest baseline*

Simplest baseline was analysing only file co-occurrences, basically it outputs predictions with higher co-occurence with input file.

The results for this baseline are:
Precision: 0.1394, Recall: 0.1230

However, here I've "cheated" a bit. Since there's no training or proba predictions, I can't set a threshold to distinguish between 0 and 1 preds. So in this baseline I simply take top N-1 predictions, letting the method to know how many files are changed in GT.

Note: here and later I stack all the predictions for the val set together and then perform eval. Better way to do it - average scores per commit, didn't have time to update it.

*Intermediate baseline*

I also used LogReg on 3 features: co-occurrence, same_author, file_distance. I evaluated it in a similar fashion as above - which is rude. Took k top probabilities, where k is number of co-changed files. Scores are below.
Results:
Precision@k: 0.1526
Recall@k: 0.1620

*Complex baseline*

Here I build LightGBM classifier. Simple, yet effective. Now the co-occurrence is used as one of the features.
In this scenario

Input is vecor X, which consists of the following features:
- co-occurrence of x_i with x_j;
- binary feature showing whether the author of the child commit differs from the author of parent commit;
- distance between files - if the files are located in the same subfolder, the distance is 0;
- extension of x_i
- new_lines of x_i
- old_lines of x_i

No hyperparam tuning was performed.

**Note**
As the original task formulated simply in the form {input: list_of_files, output: list_of_files}, some of the features used in this baseline may be discarded: e.g. number of changed lines, author of the commit.

*Results*

The resulting model shows the following results on the validation set, with the THRESHOLD set to 0.9:

Precision: 0.0785
Recall: 0.1044
F1 Score: 0.0896

When evaluating per sample and then averaging the score, the metrics are
AVG Precision: 0.1934
AVG Recall: 0.1081

Note. Although these scores are lower, they are more fair than in baselines above. Didn't have time for proper comparison...


*Future work*

An interesting direction here would be to explore GNNs, more specifically, temporal-GNNs, with files as nodes, co-occurence as edges and diff/file features as node embeddings. Then the task could be handled as edge prediction
