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


def view_dataset(dataset, to_print=1):
    if to_print == 0:  # Print the set as is.
        print(dataset.values)
    elif to_print == 1:  # Default print way
        print(dataset)
    elif to_print == 2:  # Peek at the first 20 elements
        print(dataset.head(20))
    elif to_print == 3:  # Print some info of the dataset
        print(dataset.describe())
    elif to_print == 4:
        print(dataset.groupby('classLabel').count())
    elif to_print == 5:  # just print the shape.
        print("Shape = ", dataset.shape)


dataset = load_dataset()


# Drop NAs from variables of binary values
dataset = dataset.dropna(subset=['variable1', 'variable4', 'variable5', 'variable6', 'variable7'])

# Drop NAs from variables with low number of NAs
dataset = dataset.dropna(subset=['variable2', 'variable14', 'variable17'])


# Split Columns with comma (,) in its data
dataset['variable2_x'], dataset['variable2_y'] = dataset['variable2'].str.split(',', 1).str
dataset['variable3_x'], dataset['variable3_y'] = dataset['variable3'].str.split(',', 1).str
dataset['variable8_x'], dataset['variable8_y'] = dataset['variable8'].str.split(',', 1).str


# Removed Column: Variable18 which had a lot of NANs (2000+)
# Rearrange columns after modifications
cols = ['variable1', 'variable2_x', 'variable2_y', 'variable3_x', 'variable3_y', 'variable4', 'variable5', 'variable6', 'variable7', 'variable8_x', 'variable8_y', 'variable9', 'variable10', 'variable11', 'variable12', 'variable13', 'variable14', 'variable15', 'variable17', 'variable19', 'classLabel']
dataset = dataset[cols]


# Remove NAs from columns with # of NAs of 100+
dataset['variable2_y'] = dataset.variable2_y.astype(float)
dataset['variable2_y'] = dataset['variable2_y'].fillna((dataset['variable2_y'].mean()))

dataset['variable3_y'] = dataset.variable3_y.astype(float)
dataset['variable3_y'] = dataset['variable3_y'].fillna((dataset['variable3_y'].mean()))

dataset['variable8_y'] = dataset.variable8_y.astype(float)
dataset['variable8_y'] = dataset['variable8_y'].fillna((dataset['variable8_y'].mean()))


# Count NANs in each column
count_nan = len(dataset) - dataset.count()
print(count_nan)

# Display the Data
pandas.set_option('display.expand_frame_repr', False)
view_dataset(dataset, 1)


# print(dataset['variable4'].unique())
# print(dataset['variable2.x'].value_counts())
# print(dataset.count())

