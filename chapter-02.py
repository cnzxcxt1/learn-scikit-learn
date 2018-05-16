#import sklearn as sk
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces

faces = fetch_olivetti_faces()
print(faces.DESCR)

print(faces.keys())

print(faces.images.shape)

print(faces.data.shape)

print(faces.target.shape)

print(np.max(faces.data))

print(np.min(faces.data))

print(np.mean(faces.data))

def print_faces(images, target, top_n):
    # set up the figure size in inches
    fig = plt.figure(figsize=(12, 12))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
    for i in range(top_n):
        # plot the images in a matrix of 20x20
        p = fig.add_subplot(20, 20, i + 1, xticks=[], yticks=[])
        p.imshow(images[i], cmap=plt.cm.bone)
        # label the image with the target value
        p.text(0, 14, str(target[i]))
        p.text(0, 60, str(i))

print_faces(faces.images, faces.target, 20)

plt.show()

# Training a Support Vector Machine
from sklearn.svm import SVC
svc_1 = SVC(kernel='linear')
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(faces.data, faces.target, test_size=0.25, random_state=0)

#from sklearn.cross_validation import cross_val_score, KFold
#from scipy.stats import sem
def evaluate_cross_validation(clf, X, y, K):
    # create a k-fold croos validation iterator
    cv = KFold(len(y), K, shuffle=True, random_state=0)
    # by default the score used is the one returned by score method of the estimator (accuracy)
    scores = cross_val_score(clf, X, y, cv=cv)
    print(scores)
    print(("Mean score: {0:.3f} (+/-{1:.3f})").format(np.mean(scores), sem(scores)))

evaluate_cross_validation(svc_1, X_train, y_train, 5)



#from sklearn import metrics

def train_and_evaluate(clf, X_train, X_test, y_train, y_test):
    clf.fit(X_train, y_train)
    print("Accuracy on training set:")
    print(clf.score(X_train, y_train))
    print("Accuracy on testing set:")
    print(clf.score(X_test, y_test))
    y_pred = clf.predict(X_test)

    print("Classification Report:")
    print(metrics.classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(metrics.confusion_matrix(y_test, y_pred))

train_and_evaluate(svc_1, X_train, X_test, y_train, y_test)

# the index ranges of images of people with glasses
glasses = [
    (10, 19), (30, 32), (37, 38), (50, 59), (63, 64),
    (69, 69), (120, 121), (124, 129), (130, 139), (160, 161),
    (164, 169), (180, 182), (185, 185), (189, 189), (190, 192),
    (194, 194), (196, 199), (260, 269), (270, 279), (300, 309),
    (330, 339), (358, 359), (360, 369)
]


def create_target(segments):
    # create a new y array of target size initialized with zeros
    y = np.zeros(faces.target.shape[0])
    # put 1 in the specified segments
    for (start, end) in segments:
        y[start:end + 1] = 1
    return y

target_glasses = create_target(glasses)

X_train, X_test, y_train, y_test = train_test_split(faces.data, target_glasses, test_size=0.25, random_state=0)
svc_2 = SVC(kernel='linear')

evaluate_cross_validation(svc_2, X_train, y_train, 5)

train_and_evaluate(svc_2, X_train, X_test, y_train, y_test)


X_test = faces.data[30:40]
y_test = target_glasses[30:40]
print(y_test.shape[0])
#10
select = np.ones(target_glasses.shape[0])
select[30:40] = 0
X_train = faces.data[select == 1]
y_train = target_glasses[select == 1]
print(y_train.shape[0])
#390
svc_3 = SVC(kernel='linear')
train_and_evaluate(svc_3, X_train, X_test, y_train, y_test)

y_pred = svc_3.predict(X_test)

eval_faces = [np.reshape(a, (64, 64)) for a in X_test]

print_faces(eval_faces, y_pred, 10)
plt.show()


from sklearn.datasets import fetch_20newsgroups
news = fetch_20newsgroups(subset='all')

print(type(news.data), type(news.target), type(news.target_names))

print(news.target_names)

print(len(news.data))
print(len(news.target))

print(news.data[0])

print(news.target[0], news.target_names[news.target[0]])


SPLIT_PERC = 0.75
split_size = int(len(news.data)*SPLIT_PERC)
X_train = news.data[:split_size]
X_test = news.data[split_size:]
y_train = news.target[:split_size]
y_test = news.target[split_size:]

from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer, CountVectorizer

clf_1 = Pipeline([
    ('vect', CountVectorizer()),
    ('clf', MultinomialNB()),
])

clf_2 = Pipeline([
    ('vect', HashingVectorizer(non_negative=True)),
    ('clf', MultinomialNB()),
])

clf_3 = Pipeline([
    ('vect', TfidfVectorizer()),
    ('clf', MultinomialNB()),
])

#from sklearn.cross_validation import cross_val_score, KFold
#from scipy.stats import sem

def evaluate_cross_validation(clf, X, y, K):
    # create a k-fold croos validation iterator of k=5 folds
    cv = KFold(len(y), K, shuffle=True, random_state=0)
    # by default the score used is the one returned by score method of the estimator (accuracy)
    scores = cross_val_score(clf, X, y, cv=cv)
    print(scores)
    print(("Mean score: {0:.3f} (+/-{1:.3f})").format(np.mean(scores), sem(scores)))

clfs = [clf_1, clf_2, clf_3]

for clf in clfs:
    evaluate_cross_validation(clf, news.data, news.target, 5)



clf_4 = Pipeline([
    ('vect', TfidfVectorizer(
        token_pattern=r"\b[a-z0-9_\-\.]+[a-z][a-z0-9_\-\.]+\b",
    )),
    ('clf', MultinomialNB()),
])
evaluate_cross_validation(clf_4, news.data, news.target, 5)

def get_stop_words():
    result = set()
    for line in open('stopwords_en.txt', 'r').readlines():
        result.add(line.strip())
    return result

clf_5 = Pipeline([
    ('vect', TfidfVectorizer(
        stop_words= get_stop_words(),
        token_pattern= r"\b[a-z0-9_\-\.]+[a-z][a-z0-9_\-\.]+\b",
    )),
    ('clf', MultinomialNB()),
])
evaluate_cross_validation(clf_5, news.data, news.target, 5)


clf_6 = Pipeline([
    ('vect', TfidfVectorizer(
        stop_words= get_stop_words(),
        token_pattern= r"\b[a-z0-9_\-\.]+[a-z][a-z0-9_\-\.]+\b",
    )),
    ('clf', MultinomialNB(alpha = 0.01)),
])

evaluate_cross_validation(clf_6, news.data, news.target, 5)

# Evaluating the performance

from sklearn import metrics
def train_and_evaluate(clf, X_train, X_test, y_train, y_test):
    clf.fit(X_train, y_train)
    print("Accuracy on training set:")
    print(clf.score(X_train, y_train))
    print("Accuracy on testing set:")
    print(clf.score(X_test, y_test))
    y_pred = clf.predict(X_test)

    print("Classification Report:")
    print(metrics.classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(metrics.confusion_matrix(y_test, y_pred))

train_and_evaluate(clf_6, X_train, X_test, y_train, y_test)

print(len(clf_6.named_steps['vect'].get_feature_names()))
clf_6.named_steps['vect'].get_feature_names()




# Explaining Titanic hypothesis with decision trees
# 在处理文本问题的时候，如果遇到了类似 “IndexError: list index out of range” 的问题
# 最好在搞明白他是按照什么符号进行分词的。
import csv
import numpy as np
with open('titanic.csv') as csvfile:
    titanic_reader = csv.reader(csvfile, delimiter=',', quotechar='"')
    # Header contains feature names
    row = next(titanic_reader)
    #print(row)
    feature_names = np.array(row)
    # Load dataset, and target classes
    titanic_X, titanic_y = [], []
    for row in titanic_reader:
        titanic_X.append(row)
        titanic_y.append(row[2])
        # The target value is "survived"
    titanic_X = np.array(titanic_X)
    titanic_y = np.array(titanic_y)

print(feature_names)
print(titanic_X[0], titanic_y[0])

# we keep class, age and sex
titanic_X = titanic_X[:, [1, 4, 10]]
feature_names = feature_names[[1, 4, 10]]

print(feature_names)
print(titanic_X[12],titanic_y[12])

# We have missing values for age
# Assign the mean value
ages = titanic_X[:, 1]
mean_age = np.mean(titanic_X[ages != 'NA',1].astype(np.float))
titanic_X[titanic_X[:, 1] == 'NA', 1] = mean_age




# Encode sex
from sklearn.preprocessing import LabelEncoder
enc = LabelEncoder()
label_encoder = enc.fit(titanic_X[:, 2])
print("Categorical classes:", label_encoder.classes_)
integer_classes = label_encoder.transform(label_encoder.classes_)
print("Integer classes:", integer_classes)
t = label_encoder.transform(titanic_X[:, 2])
titanic_X[:, 2] = t

print(feature_names)
print(titanic_X[12], titanic_y[12])


#
from sklearn.preprocessing import OneHotEncoder
enc = LabelEncoder()
label_encoder = enc.fit(titanic_X[:, 0])
print("Categorical classes:", label_encoder.classes_)
#Categorical classes: ['1st' '2nd' '3rd']

integer_classes = label_encoder.transform(label_encoder.classes_).reshape(3, 1)
print("Integer classes:", integer_classes)
#Integer classes: [[0] [1] [2]]

enc = OneHotEncoder()
one_hot_encoder = enc.fit(integer_classes)
# First, convert classes to 0-(N-1) integers using label_encoder
num_of_rows = titanic_X.shape[0]
t = label_encoder.transform(titanic_X[:, 0]).reshape(num_of_rows, 1)

# Second, create a sparse matrix with three columns, each one
# indicating if the instance belongs to the class
new_features = one_hot_encoder.transform(t)
# Add the new features to titanix_X
titanic_X = np.concatenate([titanic_X, new_features.toarray()], axis = 1)

#Eliminate converted columns
titanic_X = np.delete(titanic_X, [0], 1)
# Update feature names
feature_names = ['age', 'sex', 'first_class', 'second_class', 'third_class']
# Convert to numerical values
titanic_X = titanic_X.astype(np.float)
titanic_y = titanic_y.astype(np.float)

# 需要注意的是，在缺失项存在的情况下，astype 这个命令是会出错的
print(titanic_X[0], titanic_y[0])


# Training a decision tree classifier

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(titanic_X, titanic_y, test_size=0.25, random_state=33)
#Now, we can create a new DecisionTreeClassifier and

from sklearn import tree
clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=3, min_samples_leaf=5)
clf = clf.fit(X_train, y_train)



import pydotplus
from io import StringIO
dot_data = StringIO()
tree.export_graphviz(clf, out_file=dot_data, feature_names=['age','sex','1st_class','2nd_class', '3rd_class'])
#graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('titanic.png')

from IPython.core.display import Image
Image(filename='titanic.png')
#plt.show()

from sklearn import metrics
def measure_performance(X,y,clf, show_accuracy=True, show_classification_report=True, show_confussion_matrix=True):
    y_pred=clf.predict(X)
    if show_accuracy:
        print("Accuracy:{0:.3f}".format(
            metrics.accuracy_score(y, y_pred)
        ),"\n")
    if show_classification_report:
        print("Classification report")
        print(metrics.classification_report(y,y_pred),"\n")
    if show_confussion_matrix:
        print("Confussion matrix")
        print(metrics.confusion_matrix(y,y_pred),"\n")

measure_performance(X_train,y_train,clf, show_classification_report = False, show_confussion_matrix = False)


from sklearn.cross_validation import cross_val_score, LeaveOneOut
from scipy.stats import sem
def loo_cv(X_train, y_train,clf):
    # Perform Leave-One-Out cross validation
    # We are preforming 1313 classifications!
    loo = LeaveOneOut(X_train[:].shape[0])
    scores = np.zeros(X_train[:].shape[0])
    for train_index, test_index in loo:
        X_train_cv, X_test_cv = X_train[train_index], X_train[test_index]
        y_train_cv, y_test_cv = y_train[train_index], y_train[test_index]
        clf = clf.fit(X_train_cv,y_train_cv)
        y_pred = clf.predict(X_test_cv)
        scores[test_index] = metrics.accuracy_score(y_test_cv.astype(int), y_pred.astype(int))
    print(("Mean score: {0:.3f} (+/-{1:.3f})").format(np.mean(scores), sem(scores)))
    #print(np.mean(scores),sem(scores))
loo_cv(X_train, y_train,clf)


# Random Forests – randomizing decisions 随机森林

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=10, random_state=33)
clf = clf.fit(X_train, y_train)
loo_cv(X_train, y_train, clf)

clf_dt = tree.DecisionTreeClassifier(criterion='entropy', max_depth=3, min_samples_leaf=5)
clf_dt.fit(X_train, y_train)
measure_performance(X_test, y_test, clf_dt)
#Accuracy:0.793


#Predicting house prices with regression

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
boston = load_boston()
print(boston.data.shape)
#(506, 13)
print(boston.feature_names)
#['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO' 'B' 'LSTAT' 'MEDV']
print(np.max(boston.target), np.min(boston.target), np.mean(boston.target))
# 50.0 5.0 22.5328063241

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(boston.data, boston.target, test_size=0.25, random_state=33)
from sklearn.preprocessing import StandardScaler
scalerX = StandardScaler().fit(X_train)
#scalery = StandardScaler().fit(y_train)
X_train = scalerX.transform(X_train)
#y_train = scalery.transform(y_train)
X_test = scalerX.transform(X_test)
#y_test = scalery.transform(y_test)


from sklearn.cross_validation import *
def train_and_evaluate(clf, X_train, y_train):
    clf.fit(X_train, y_train)
    print("Coefficient of determination on training set:",clf.score(X_train, y_train))
    # create a k-fold cross validation iterator of k=5 folds
    cv = KFold(X_train.shape[0], 5, shuffle=True, random_state=33)
    scores = cross_val_score(clf, X_train, y_train, cv=cv)
    print("Average coefficient of determination using 5-fold crossvalidation:",np.mean(scores))

from sklearn import linear_model
clf_sgd = linear_model.SGDRegressor(loss='squared_loss', penalty=None, random_state=42)
train_and_evaluate(clf_sgd,X_train,y_train)
#Coefficient of determination on training set: 0.743303511411
#Average coefficient of determination using 5-fold crossvalidation: 0.715166411086

print(clf_sgd.coef_)


clf_sgd1 = linear_model.SGDRegressor(loss='squared_loss', penalty='l2', random_state=42)
train_and_evaluate(clf_sgd1, X_train, y_train)


# Second try – Support Vector Machines for regression
from sklearn import svm
clf_svr = svm.SVR(kernel='linear')
train_and_evaluate(clf_svr, X_train, y_train)

clf_svr_poly = svm.SVR(kernel='poly')
train_and_evaluate(clf_svr_poly, X_train, y_train)

clf_svr_rbf = svm.SVR(kernel='rbf')
train_and_evaluate(clf_svr_rbf, X_train, y_train)

# Third try – Random Forests revisited
from sklearn import ensemble
import numpy
clf_et=ensemble.ExtraTreesRegressor(n_estimators=10, random_state=42)
train_and_evaluate(clf_et, X_train, y_train)

print(sorted(zip(clf_et.feature_importances_, boston.feature_names)))

from sklearn import metrics
def measure_performance(X, y, clf, show_accuracy=True, show_classification_report=True, show_confusion_matrix=True, show_r2_score=False):
    y_pred = clf.predict(X)
    if show_accuracy:
        print("Accuracy:{0:.3f}".format(metrics.accuracy_score(y, y_pred)),"\n")
    if show_classification_report:
        print("Classification report")
        print(metrics.classification_report(y, y_pred),"\n")
    if show_confusion_matrix:
        print("Confusion matrix")
        print(metrics.confusion_matrix(y, y_pred),"\n")
    if show_r2_score:
        print("Coefficient of determination:{0:.3f}".format(metrics.r2_score(y, y_pred) ),"\n")

measure_performance(X_test, y_test, clf_et, show_accuracy=False,
                    show_classification_report=False,
                    show_confusion_matrix=False,
                    show_r2_score=True)

