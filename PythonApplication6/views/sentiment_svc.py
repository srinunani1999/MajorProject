from __future__ import division
import collections
import random 
import numpy as np
from sklearn import svm
from sklearn.metrics import f1_score
import pandas as pd

def number_of_chars(s):
    return len(s)

def unique_chars(s):
    s2 = ''.join(set(s))
    return len(s2)

def weighted_unique_chars(s):
    return unique_chars(s)/number_of_chars(s)

def words_count(s):
    return collections.Counter(s)

def words_counter_object(s):
    cnt = collections.Counter()

    words = s.split()
    for w in words:
        cnt[w] += 1

    return cnt

def total_words(cnt):
    sum = 0
    for k in dict(cnt).keys():
        sum += int(cnt[k])

    return sum

def is_repeated(cnt):
    for k,v in cnt.most_common(1):
        freq = v/total_words(cnt)
        if freq > 0.5:
            return 1
    return 0

def discrete_label(target_value):
    if target_value != float(1):
        target_value = 0
    return target_value

def make_feature_vector(critique, labels):
    " construct feature vector" 
    feature_vector = []
    for i in range(len(critique)):
        s = critique[i]
        feature = []
        counter_obj = words_counter_object(s)

        feature.append(number_of_chars(s))
        feature.append(unique_chars(s))
        feature.append(weighted_unique_chars(s))
        feature.append(total_words(counter_obj))
        feature.append(is_repeated(counter_obj))

        feature.append(discrete_label(labels[i]))
        feature_vector.append(feature)

    return feature_vector

def read_data():
    ''' read and make a list of critiques'''
    df = pd.read_csv('views/data/Evaluations-Binary.csv')
    critiques,labels = df['Review'].tolist(), df['Useful'].tolist()
    return critiques,labels

def make_np_array_XY(xy):
    a = np.array(xy)
    x = a[:,0:-1]
    y = a[:,-1]
    return x,y

def train_model():
    critiques, labels = read_data()
    features_and_labels = make_feature_vector(critiques, labels)
    number_of_features = len(features_and_labels[0]) - 1
    random.shuffle(features_and_labels)

    mid_index = int(len(features_and_labels)/2)
    XY_train = features_and_labels[:mid_index]
    XY_test = features_and_labels[mid_index:]

    X_train, Y_train = make_np_array_XY(XY_train)
    X_test, Y_test = make_np_array_XY(XY_test)

    global svc
    C = 1.0
    svc = svm.SVC(kernel='linear', C=C).fit(X_train, Y_train)


def predict_text(text):
    x_test = [[]]
    counter_obj = words_counter_object(text)
    x_test[0].append(number_of_chars(text))
    x_test[0].append(unique_chars(text))
    x_test[0].append(weighted_unique_chars(text))
    x_test[0].append(total_words(counter_obj))
    x_test[0].append(is_repeated(counter_obj))

    y_predict = svc.predict(x_test)
    print(y_predict)
    return "positive" if y_predict[0] == 1 else "negative"

svc = None
