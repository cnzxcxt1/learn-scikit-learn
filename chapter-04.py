# %pylab inline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

titanic = pd.read_csv('titanic.csv')
print(titanic)

print(titanic.head()[['pclass', 'survived', 'age', 'embarked', 'boat', 'sex']])

from sklearn import feature_extraction
def one_hot_dataframe(data, cols, replace=False):
    vec = feature_extraction.DictVectorizer()
    mkdict = lambda row: dict((col, row[col]) for col in cols)
    vecData = pd.DataFrame(vec.fit_transform( data[cols].apply(mkdict, axis=1)).toarray())
    vecData.columns = vec.get_feature_names()
    vecData.index = data.index
    if replace:
        data = data.drop(cols, axis=1)
        data = data.join(vecData)
    return data,vecData


titanic,titanic_n = one_hot_dataframe(titanic, ['pclass', 'embarked', 'sex'], replace=True)
titanic.describe()

titanic, titanic_n = one_hot_dataframe(titanic, ['home.dest', 'room', 'ticket', 'boat'], replace=True)

mean = titanic['age'].mean()
titanic['age'].fillna(mean, inplace=True)
titanic.fillna(0, inplace=True)


from sklearn.cross_validation import train_test_split
titanic_target = titanic['survived']
titanic_data = titanic.drop(['name', 'row.names', 'survived'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(titanic_data, titanic_target, test_size=0.25, random_state=33)


from sklearn import tree
dt = tree.DecisionTreeClassifier(criterion='entropy')
dt = dt.fit(X_train, y_train)

from sklearn import metrics
y_pred = dt.predict(X_test)
print("Accuracy:{0:.3f}".format(metrics.accuracy_score(y_test, y_pred)), "\n")



# Feature selection
print(titanic)

from sklearn import feature_selection
fs = feature_selection.SelectPercentile( feature_selection.chi2, percentile=20)
X_train_fs = fs.fit_transform(X_train, y_train)

dt.fit(X_train_fs, y_train)
X_test_fs = fs.transform(X_test)
y_pred_fs = dt.predict(X_test_fs)
print("Accuracy:{0:.3f}".format(metrics.accuracy_score(y_test, y_pred_fs)),"\n")

from sklearn import cross_validation
percentiles = range(1, 100, 5)
results = []
for i in range(1,100,5):
    fs = feature_selection.SelectPercentile( feature_selection.chi2, percentile=i)
    X_train_fs = fs.fit_transform(X_train, y_train)
    scores = cross_validation.cross_val_score(dt, X_train_fs, y_train, cv=5)
    # print i,scores.mean()
    results = np.append(results, scores.mean())

optimal_percentil = np.where(results == results.max( ))[0][0]
print("Optimal number of features:{0}".format(percentiles[optimal_percentil]), "\n")
#plt.show()
#print(optimal_percentil)
#Optimal number of features:11
# Plot number of features VS. cross-validation scores
import pylab as pl
pl.figure()
pl.xlabel("Number of features selected")
pl.ylabel("Cross-validation accuracy)")
pl.plot(percentiles, results)
plt.show()


fs = feature_selection.SelectPercentile( feature_selection.chi2, percentile=percentiles[optimal_percentil])
X_train_fs = fs.fit_transform(X_train, y_train)
dt.fit(X_train_fs, y_train)
X_test_fs = fs.transform(X_test)
y_pred_fs = dt.predict(X_test_fs)
print("Accuracy:{0:.3f}".format(metrics.accuracy_score(y_test, y_pred_fs)), "\n")


dt = tree.DecisionTreeClassifier(criterion='entropy')
scores = cross_validation.cross_val_score(dt, X_train_fs,y_train, cv=5)
print("Entropy criterion accuracy on cv: {0:.3f}".format(scores.mean()))
# Entropy criterion accuracy on cv: 0.889
dt = tree.DecisionTreeClassifier(criterion='gini')
scores = cross_validation.cross_val_score(dt, X_train_fs, y_train, cv=5)
print("Gini criterion accuracy on cv: {0:.3f}".format(scores.mean()))

dt.fit(X_train_fs, y_train)
X_test_fs = fs.transform(X_test)
y_pred_fs = dt.predict(X_test_fs)
print("Accuracy: {0:.3f}".format(metrics.accuracy_score(y_test, y_pred_fs)),"\n")

from sklearn.datasets import fetch_20newsgroups
news = fetch_20newsgroups(subset='all')
n_samples = 3000
X_train = news.data[:n_samples]
y_train = news.target[:n_samples]

from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

def get_stop_words():
    result = set()
    for line in open('stopwords_en.txt', 'r').readlines():
        result.add(line.strip())
    return result

stop_words = get_stop_words()

clf = Pipeline([('vect',
                 TfidfVectorizer( stop_words=stop_words,
                                  token_pattern= r"\b[a-z0-9_\-\.]+[a-z][a-z0-9_\-\.]+\b",)),
                ('nb', MultinomialNB(alpha=0.01)),])

from sklearn.cross_validation import cross_val_score, KFold
from scipy.stats import sem
def evaluate_cross_validation(clf, X, y, K):
    # create a k-fold croos validation iterator of k=5 folds
    cv = KFold(len(y), K, shuffle=True, random_state=0)
    # by default the score used is the one returned by score method of the estimator (accuracy)
    scores = cross_val_score(clf, X, y, cv=cv)
    print(scores)
    print(("Mean score: {0:.3f} (+/-{1:.3f})").format(np.mean(scores), sem(scores)))

evaluate_cross_validation(clf, X_train, y_train, 3)


def calc_params(X, y, clf, param_values, param_name, K):
    # initialize training and testing scores with zeros
    train_scores = np.zeros(len(param_values))
    test_scores = np.zeros(len(param_values))

    # iterate over the different parameter values
    for i, param_value in enumerate(param_values):
        print(param_name, ' = ', param_value)
        # set classifier parameters
        clf.set_params(**{param_name:param_value})
        # initialize the K scores obtained for each fold
        k_train_scores = np.zeros(K)
        k_test_scores = np.zeros(K)
        # create KFold cross validation
        cv = KFold(n_samples, K, shuffle=True, random_state=0)
        # iterate over the K folds
        for j, (train, test) in enumerate(cv):
            clf.fit([X[k] for k in train], y[train])
            k_train_scores[j] = clf.score([X[k] for k in train], y[train])
            k_test_scores[j] = clf.score([X[k] for k in test], y[test])

        train_scores[i] = np.mean(k_train_scores)
        test_scores[i] = np.mean(k_test_scores)

    # plot the training and testing scores in a log scale
    plt.semilogx(param_values, train_scores, alpha=0.4, lw=2, c='b')
    plt.semilogx(param_values, test_scores, alpha=0.4, lw=2, c='g')
    plt.xlabel("Alpha values")
    plt.ylabel("Mean cross-validation accuracy")
    # return the training and testing scores on each parameter value
    return(train_scores, test_scores)

alphas = np.logspace(-7, 0, 8)
print(alphas)

train_scores, test_scores = calc_params(X_train, y_train, clf, alphas, 'nb__alpha', 3)

print('training scores: ', train_scores)
print('testing scores: ', test_scores)

from sklearn.svm import SVC

clf = Pipeline([
    ('vect', TfidfVectorizer( stop_words=stop_words,
                              token_pattern= r"\b[a-z0-9_\-\.]+[a-z][a-z0-9_\-\.]+\b",
                              )), ('svc', SVC()),])

gammas = np.logspace(-2, 1, 4)
train_scores, test_scores = calc_params(X_train, y_train, clf, gammas,'svc__gamma', 3)

from sklearn.grid_search import GridSearchCV
parameters = {
                 'svc__gamma': np.logspace(-2, 1, 4),
                 'svc__C': np.logspace(-1, 1, 3),
}

clf = Pipeline([
    ('vect',TfidfVectorizer(stop_words=stop_words,
                                 token_pattern= r"\b[a-z0-9_\-\.]+[a-z][a-z0-9_\-\.]+\b",
                            )),('svc', SVC()),])

gs = GridSearchCV(clf, parameters, verbose=2, refit=False, cv=3)

%time _ = gs.fit(X_train, y_train)


gs.best_params_, gs.best_score_



from sklearn.externals import joblib
from sklearn.cross_validation import ShuffleSplit
import os
def persist_cv_splits(X, y, K=3, name='data', suffix="_cv_%03d.pkl"):
    """Dump K folds to filesystem."""
    cv_split_filenames = []
    # create KFold cross validation
    cv = KFold(n_samples, K, shuffle=True, random_state=0)
    # iterate over the K folds
    for i, (train, test) in enumerate(cv):
        cv_fold = ([X[k] for k in train], y[train], [X[k] for k in test], y[test])
        cv_split_filename = name + suffix % i
        cv_split_filename = os.path.abspath(cv_split_filename)
        joblib.dump(cv_fold, cv_split_filename)
        cv_split_filenames.append(cv_split_filename)

    return cv_split_filenames

cv_filenames = persist_cv_splits(X_train, y_train, name='news')

def compute_evaluation(cv_split_filename, clf, params):
    # All module imports should be executed in the worker namespace
    from sklearn.externals import joblib

    # load the fold training and testing partitions from the filesystem
    X_train, y_train, X_test, y_test = joblib.load( cv_split_filename, mmap_mode='c')
    clf.set_params(**params)
    clf.fit(X_train, y_train)
    test_score = clf.score(X_test, y_test)
    return test_score

from sklearn.grid_search import ParameterGrid

def parallel_grid_search(lb_view, clf, cv_split_filenames, param_grid):
    all_tasks = []
    all_parameters = list(ParameterGrid(param_grid))

    # iterate over parameter combinations
    for i, params in enumerate(all_parameters):
        task_for_params = []
        # iterate over the K folds
        for j, cv_split_filename in enumerate(cv_split_filenames):
            t = lb_view.apply( compute_evaluation, cv_split_filename, clf, params)
            task_for_params.append(t)
        all_tasks.append(task_for_params)
    return all_parameters, all_tasks





### For this part, the code cannot work with parallel package
from sklearn.svm import SVC
from IPython.parallel import Client

client = Client()
lb_view = client.load_balanced_view()
all_parameters, all_tasks = parallel_grid_search(lb_view, clf, cv_filenames, parameters)

def print_progress(tasks):
    progress = np.mean([task.ready() for task_group in tasks for task in task_group])
    print("Tasks completed: {0}%".format(100 * progress))

print_progress(all_tasks)


def find_bests(all_parameters, all_tasks, n_top=5):
    """Compute the mean score of the completed tasks"""
    mean_scores = []
    for param, task_group in zip(all_parameters, all_tasks):
        scores = [t.get() for t in task_group if t.ready()]
        if len(scores) == 0:
            continue
        mean_scores.append((np.mean(scores), param))
    return sorted(mean_scores, reverse=True)[:n_top]
print(find_bests(all_parameters, all_tasks))