# Import Modules
import sys
import scipy
import numpy
import matplotlib
import sklearn
import pandas


# Check the versions of libraries
def check_versions():
    print('Python: {}'.format(sys.version))
    print('scipy: {}'.format(scipy.__version__))
    print('numpy: {}'.format(numpy.__version__))
    print('matplotlib: {}'.format(matplotlib.__version__))
    print('pandas: {}'.format(pandas.__version__))
    print('sklearn: {}'.format(sklearn.__version__))


def load_dataset():
    url_train = "training.csv"
    url_test = "validation.csv"

    train = load(url_train)
    train = train.sample(frac=1).reset_index(drop=True)

    train_x = train.drop(['classLabel'], axis=1)
    train_x = fix_encode_test(train_x, 'train')
    train_y = train['classLabel']

    test = load(url_test)
    test = test.sample(frac=1).reset_index(drop=True)

    test_x = test.drop(['classLabel'], axis=1)
    test_x = fix_encode_test(test_x, 'test')
    test_y = test['classLabel']

    return train_x, train_y, test_x, test_y


def load(url):
    names = ["variable1", "variable2", "variable3", "variable4", "variable5", "variable6", "variable7", "variable8",
             "variable9", "variable10", "variable11", "variable12", "variable13", "variable14", "variable15",
             "variable17", "variable18", "variable19", "classLabel"]
    dataset = pandas.read_csv(url, names=names, sep=";")

    dataset = clean_dataframe(dataset)
    dataset = normalize_dataframe(dataset)
    dataset = one_hot_encode(dataset)

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
    df['classLabel'] = df.pop('classLabel')
    return df


def fix_encode_test(df, distrib):
    if distrib == 'test':
        df['variable4_l'] = 0
        df['variable5_gg'] = 0
        df['variable6_r'] = 0
        df['variable7_o'] = 0
        df['variable13_p'] = 0
    elif distrib == 'train':
        df['variable4_l'] = df.pop('variable4_l')
        df['variable5_gg'] = df.pop('variable5_gg')
        df['variable6_r'] = df.pop('variable6_r')
        df['variable7_o'] = df.pop('variable7_o')
        df['variable13_p'] = df.pop('variable13_p')

    return df

# dataset = load_dataset()
# dataset = clean_dataframe(dataset)
# dataset = normalize_dataframe(dataset, 'mean')
# dataset = one_hot_encode(dataset)
#
# Display the Data
# pandas.set_option('display.expand_frame_repr', False)
# view_dataframe(dataset)

