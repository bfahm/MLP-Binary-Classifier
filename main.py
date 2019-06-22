# Import Modules
import sys
import scipy
import numpy
import matplotlib
import sklearn
import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


seed = 7  # Retrieve the same results every run
scoring = 'accuracy'  # Test model score by accuracy (in %percentage)


# Check the versions of libraries
def check_versions():
    print('Python: {}'.format(sys.version))
    print('scipy: {}'.format(scipy.__version__))
    print('numpy: {}'.format(numpy.__version__))
    print('matplotlib: {}'.format(matplotlib.__version__))
    print('pandas: {}'.format(pandas.__version__))
    print('sklearn: {}'.format(sklearn.__version__))


def load_dataset():
    """
    Loads a csv file into a dataFrame
    column = 'names'
    rows from 1 to 150.
    Takes a value from 1 to 8 to peek at the data in a special form or by printing out a graph,
    If left plank, it will just load the set.
    """

    url = "training.csv"
    names = ["variable1", "variable2", "variable3", "variable4", "variable5", "variable6", "variable7", "variable8", "variable9", "variable10", "variable11", "variable12", "variable13", "variable14", "variable15", "variable17", "variable18", "variable19", "classLabel"]
    dataset = pandas.read_csv(url, names=names, sep = ";")

    return dataset


def view_dataframe(df, to_print=1):
    if to_print == 0:  # Print the set as is.
        print(df.values)
    elif to_print == 1:  # Default print way
        print(df)
    elif to_print == 2:  # Peek at the first 20 elements
        print(df.head(20))
    elif to_print == 3:  # Print some info of the df
        print(df.describe())
    elif to_print == 4:
        print(df.groupby('classLabel').count())
    elif to_print == 5:  # just print the shape.
        print("Shape = ", df.shape)


def clean_dataframe(df):
    # Drop NAs from variables of binary values
    df = df.dropna(subset=['variable1', 'variable4', 'variable5', 'variable6', 'variable7'])

    # Drop NAs from variables with low number of NAs
    df = df.dropna(subset=['variable2', 'variable14', 'variable17'])

    # Split Columns with comma (,) in its data
    df['variable2_x'], df['variable2_y'] = df['variable2'].str.split(',', 1).str
    df['variable3_x'], df['variable3_y'] = df['variable3'].str.split(',', 1).str
    df['variable8_x'], df['variable8_y'] = df['variable8'].str.split(',', 1).str

    # Removed Column: Variable18 which had a lot of NANs (2000+)
    # Rearrange columns after modifications
    cols = ['variable1', 'variable2_x', 'variable2_y', 'variable3_x', 'variable3_y', 'variable4', 'variable5',
            'variable6', 'variable7', 'variable8_x', 'variable8_y', 'variable9', 'variable10', 'variable11',
            'variable12', 'variable13', 'variable14', 'variable15', 'variable17', 'variable19', 'classLabel']
    df = df[cols]

    # Remove NAs from columns with # of NAs of 100+
    df['variable2_y'] = df.variable2_y.astype(float)
    df['variable2_y'] = df['variable2_y'].fillna((df['variable2_y'].mean()))

    df['variable3_y'] = df.variable3_y.astype(float)
    df['variable3_y'] = df['variable3_y'].fillna((df['variable3_y'].mean()))

    df['variable8_y'] = df.variable8_y.astype(float)
    df['variable8_y'] = df['variable8_y'].fillna((df['variable8_y'].mean()))

    col_fix_dtype = ['variable2_x', 'variable2_y', 'variable3_x', 'variable3_y', 'variable8_x', 'variable8_y', 'variable11', 'variable14', 'variable15', 'variable17', 'variable19']

    for i in col_fix_dtype:
        df[i] = df[i].astype(float)

    # Count NANs in each column
    # count_nan = len(df) - df.count()
    # print(count_nan)

    return df


def normalize_dataframe(df, type='mean'):
    col_to_normalize = ['variable2_x', 'variable2_y', 'variable3_x', 'variable3_y', 'variable8_x', 'variable8_y',
                     'variable11', 'variable14', 'variable15', 'variable17', 'variable19']
    if type == 'mean':
        for i in col_to_normalize:
            df[i] = (df[i] - df[i].mean()) / df[i].std()
        return df
    elif type == 'minmax':
        for i in col_to_normalize:
            df[i] = (df[i] - df[i].min()) / (df[i].max() - df[i].min())
        return df


def one_hot_encode(df):
    cols_to_one_hot = ['variable1', 'variable4', 'variable5', 'variable6', 'variable7', 'variable9', 'variable10', 'variable12', 'variable13']
    for i in cols_to_one_hot:
        dummies = pandas.get_dummies(df[i], prefix=i, drop_first=False)
        df = pandas.concat([df, dummies], axis=1)
        df = df.drop([i], axis=1)

    df['classLabel'] = df['classLabel'].map({'yes.': 1, 'no.': 0})

    # Rearrange columns
    col = ['variable2_x', 'variable2_y', 'variable3_x', 'variable3_y',
       'variable8_x', 'variable8_y', 'variable11', 'variable14', 'variable15',
       'variable17', 'variable19', 'variable1_a', 'variable1_b',
       'variable4_l', 'variable4_u', 'variable4_y', 'variable5_g',
       'variable5_gg', 'variable5_p', 'variable6_W', 'variable6_aa',
       'variable6_c', 'variable6_cc', 'variable6_d', 'variable6_e',
       'variable6_ff', 'variable6_i', 'variable6_j', 'variable6_k',
       'variable6_m', 'variable6_q', 'variable6_r', 'variable6_x',
       'variable7_bb', 'variable7_dd', 'variable7_ff', 'variable7_h',
       'variable7_j', 'variable7_n', 'variable7_o', 'variable7_v',
       'variable7_z', 'variable9_f', 'variable9_t', 'variable10_f',
       'variable10_t', 'variable12_f', 'variable12_t', 'variable13_g',
       'variable13_p', 'variable13_s', 'classLabel']
    df = df[col]
    return df


dataset = load_dataset()
dataset = clean_dataframe(dataset)
dataset = normalize_dataframe(dataset, 'mean')
dataset = one_hot_encode(dataset)

# Display the Data
pandas.set_option('display.expand_frame_repr', False)
view_dataframe(dataset, 1)

