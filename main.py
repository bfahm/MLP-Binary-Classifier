from data import *
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import random

# seed = 7  # Retrieve the same results every run
# scoring = 'accuracy'  # Test model score by accuracy (in %percentage)


def logistic_model(X_train, X_test, Y_train):
    classifier = LogisticRegression(solver='liblinear', multi_class='ovr')
    classifier.fit(X_train, Y_train)
    predictions = classifier.predict(X_test)
    return classifier, predictions


def neural_model(X_train, X_test, Y_train):
    classifier = MLPClassifier(solver='lbfgs', alpha=0.1, hidden_layer_sizes=(5, 5, 5, 2), random_state=1)
    classifier.fit(X_train, Y_train)
    predictions = classifier.predict(X_test)
    return classifier, predictions

def print_model_specs(predictions, Y_test):
    print("Accuracy is: ", accuracy_score(Y_test, predictions))
    #print("------------------------------------------------------------------")
    #print(classification_report(Y_test, predictions))


def drop_features(train_x, test_x, needed):
    all_columns = ['variable2_x', 'variable2_y', 'variable3_x', 'variable3_y',
       'variable8_x', 'variable8_y', 'variable11', 'variable14', 'variable15',
       'variable17', 'variable19', 'variable1_a', 'variable1_b', 'variable4_u',
       'variable4_y', 'variable5_g', 'variable5_p', 'variable6_W',
       'variable6_aa', 'variable6_c', 'variable6_cc', 'variable6_d',
       'variable6_e', 'variable6_ff', 'variable6_i', 'variable6_j',
       'variable6_k', 'variable6_m', 'variable6_q', 'variable6_x',
       'variable7_bb', 'variable7_dd', 'variable7_ff', 'variable7_h',
       'variable7_j', 'variable7_n', 'variable7_v', 'variable7_z',
       'variable9_f', 'variable9_t', 'variable10_f', 'variable10_t',
       'variable12_f', 'variable12_t', 'variable13_g', 'variable13_s',
       'variable4_l', 'variable5_gg', 'variable6_r', 'variable7_o',
       'variable13_p']
    selected_features = []
    selected_features_names = []
    ret_train_x = pandas.DataFrame()
    ret_test_x = pandas.DataFrame()
    for x in range(needed):
        selected_features.append(random.randint(0, 50))

    selected_features = list(dict.fromkeys(selected_features))
    for i in selected_features:
        ret_train_x[all_columns[i]] = train_x[all_columns[i]]
        ret_test_x[all_columns[i]] = test_x[all_columns[i]]
        selected_features_names.append(all_columns[i])

    return ret_train_x, ret_test_x, selected_features_names


def choose_features(train_x, test_x, features):
    ret_train_x = pandas.DataFrame()
    ret_test_x = pandas.DataFrame()

    for i in features:
        ret_train_x[i] = train_x[i]
        ret_test_x[i] = test_x[i]

    return ret_train_x, ret_test_x


def print_raw():
    #pandas.set_option('display.expand_frame_repr', False)
    print(train_x.columns)
    #print(train_y)
    print(test_x.columns)
    #print(test_y)


train_x, train_y, test_x, test_y = load_dataset()
#train_x, test_x, selected_features = drop_features(train_x, test_x, 100)
chosen = ['variable6_d', 'variable14', 'variable7_bb', 'variable4_u', 'variable6_ff', 'variable10_f', 'variable13_g', 'variable11', 'variable6_i', 'variable6_m', 'variable2_x', 'variable9_t', 'variable7_z', 'variable9_f', 'variable7_ff', 'variable6_x', 'variable3_x', 'variable6_c', 'variable1_a', 'variable7_n', 'variable5_g', 'variable12_f', 'variable3_y', 'variable4_y', 'variable7_dd', 'variable1_b', 'variable7_h', 'variable6_cc', 'variable6_j', 'variable7_j']
train_x, test_x = choose_features(train_x, test_x, chosen)
classifier, predictions = neural_model(train_x, test_x, train_y)
print_model_specs(predictions, test_y)
#print(selected_features)

#print_raw()