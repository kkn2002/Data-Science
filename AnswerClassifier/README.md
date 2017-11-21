# Quora Answer Classifier

Quora uses a combination of machine learning (ML) algorithms and moderation to ensure high-quality content on the site. High answer quality has helped Quora distinguish itself from other Q&A sites on the web.  

Your task will be to devise a classifier that is able to tell good answers from bad answers, as well as humans can.  A good answer is denoted by a +1 in our system, and a bad answer is denoted by a -1.

Input format (read from STDIN):
The first line contains N, M. N = Number of training data records, M = number of parameters. Followed by N lines containing records of training data. Then one integer q, q = number of records to be classified, followed by q lines of query data

Training data corresponds to the following format:
<answer-identifier> <+1 or -1> (<feature-index>:<feature-value>)*

Query data corresponds to the following format:
<answer-identifier> (<feature-index>:<feature-value>)*

The answer identifier  is an alphanumeric string of no more than 10 chars.  Each identifier is guaranteed unique.  All feature values are doubles.
0 < M < 100
0 < N < 50,000
0 < q < 5,000

This data is completely anonymized and extracted from real production data, and thus will not include the raw form of the
answers. We, however, have extracted as many features as we think are useful, and you can decide which features make sense to be included in your final algorithm. The actual labeling of a good answer and bad answer is done organically on our site, through human moderators.

Output format (write to STDOUT):
For each query, you should output q lines to stdout, representing the decision made by your classifier, whether each answer is good or not:

<answer-identifier> <+1 or -1>

You are given a relative large sample input dataset offline with its corresponding output to finetune your program with your ML libraries.  It can be downloaded here: http://qsf.cf.quoracdn.net/Quora...

# Scoring
Only one very large test dataset will be given for this problem online as input to your program for scoring.  This input data set will not be revealed to you.

Output for every classification is awarded points separately. The score for this problem will be the sum of points for each correct classification. To prevent naive solution credit (outputting all +1s, for example), points are awarded only after X correct classifications, where X is number of +1 answers or -1 answers (whichever is greater).

# Timing
Your program should complete in minutes. Try to achieve as high an accuracy as possible with this constraint in mind.
