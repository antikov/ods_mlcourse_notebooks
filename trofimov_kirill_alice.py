import warnings
import os
import numpy as np
import pandas as pd
import pickle
from collections import Counter
from scipy.sparse import csr_matrix, hstack
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.feature_extraction.text import TfidfVectorizer

def write_to_submission_file(predicted_labels, out_file,
                             target='target', index_label="session_id"):
    predicted_df = pd.DataFrame(predicted_labels,
                                index = np.arange(1, predicted_labels.shape[0] + 1),
                                columns=[target])
    predicted_df.to_csv(out_file, index_label=index_label)

print("Loading data...")

PATH_TO_DATA = ('../../data')
train_df = pd.read_csv(os.path.join(PATH_TO_DATA, 'train_sessions.csv'), index_col='session_id')
test_df = pd.read_csv(os.path.join(PATH_TO_DATA, 'test_sessions.csv'), index_col='session_id')

train_df.sample(frac=1).reset_index(drop=True)
y = train_df['target']
ratio = 0.9
idx = int(round(train_df.shape[0] * ratio))
skf = StratifiedKFold(n_splits=7, shuffle=True, random_state=17)
train_split = train_df.shape[0]


X = pd.concat([train_df.drop(columns='target'), test_df], axis = 0)

print('Extracting features...')
times = ['time%s' % i for i in range(1, 11)]
sites = ['site%s' % i for i in range(1, 11)]
X[times] = X[times].apply(pd.to_datetime)
X[sites] = X[sites].fillna(0).astype('int')
X['sites'] = X[sites].apply(lambda x: " ".join(x.astype('str')), axis = 1)

X['hour'] = X['time1'].apply(lambda x: x.hour)
X['month'] = X['time1'].apply(lambda x: x.month)
X['year'] = X['time1'].apply(lambda x: x.year)
X['yearmonth'] = 12*(X['year'] - 2013) + X['month']
X['hour1618'] = X['time1'].apply(lambda x: 16<=x.hour<=18).astype('int')
X['hour1213'] = X['time1'].apply(lambda x: 12<=x.hour<=13).astype('int')
X['hour915'] = X['time1'].apply(lambda x: (x.hour == 9 or x.hour == 15)).astype('int')
X['minute'] = X['time1'].apply(lambda x: x.minute)
X['len'] = (X['time10'] - X['time1']).apply(lambda x: np.log1p(x.total_seconds())).fillna(0)
X['hourminute'] = X['hour'] * 60 + X['minute']
X['weekday'] = X['time1'].apply(lambda x: x.weekday())

def get_part_day(x):
    h = x.hour
    if 0 <= h <= 5:
        return 0
    if 6 <= h <= 11:
        return 1
    if 12 <= h <= 17:
        return 2
    if 18 <= h <= 23:
        return 3

X['partday'] = X['time1'].apply(lambda x: get_part_day(x))

X['saturday'] = (X['weekday'] == 5).astype('int')
X['sunday'] = (X['weekday'] == 6).astype('int')
X['week'] = X['time1'].apply(lambda x: x.isocalendar()[1])


print('Creating bag of words...')

full_sites = X[sites]
sites_flatten = full_sites.values.flatten()
full_sites_sparse = csr_matrix(([1] * sites_flatten.shape[0],
                                sites_flatten,
                                range(0, sites_flatten.shape[0]  + 10, 10)))[:, 1:]


features_ohe = ['hour', 'weekday', 'partday', 'month', 'week', 'yearmonth']
ohe = OneHotEncoder().fit_transform(X[features_ohe])

train_df = X[:train_split]
test_df = X[train_split:]


print('Extracting tf idf features...')

tf = TfidfVectorizer(ngram_range=(1,7), max_features = 200000)
tf.fit(train_df['sites'].values)
tf_idf = tf.transform(X['sites'])


features = ['saturday', 'sunday', 'len', 'hour1618', 'hour1213', 'hour915']
full_df = hstack([tf_idf, X[features], ohe], format='csr')
X_train = full_df[:train_split]
X_test = full_df[train_split:]


print('Learning logistic regression...')

linear = LogisticRegression(C=5)
linear.fit(X_train, y)
test_pred = linear.predict_proba(X_test)[:, 1]
print('Predicted alice sessions:', sum(linear.predict(X_test)))

print('Creating assignment6_alice_submission.csv...')
write_to_submission_file(test_pred, "assignment6_alice_submission.csv")