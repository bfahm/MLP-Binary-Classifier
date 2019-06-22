# Binary Classification Problem

### Project Dependencies:

- Python 3.7
- Pandas
- SKLearn

*Given a training data of about 3900 examples (training and validation), it is required to clean the data, and create a model that classifies the validation data correctly.*

## Thoughts and observations

**First: Data Cleaning** - `data.py`

I dealt with NANs with two aproaches:

- For Columns containing descrete values, NANs was unfeasible, so I removed rows containing them.
- For Columns containing non-descrete values
	- If the column had low number of NANs (<100), I completely removed it
	- If the column had high number of NANs (>100), I replaced it with the mean value of that column.
- There was a column that had a lot of NANs (~2000) which made it not useful at all, so I dropped it completely.


Next, I "hot encoded" columns with descrete values, extracting the dummies to new columns and deleting the old column.
At that point, columns order got messed up, but since that wouldn't impact the model, I discarded reordering it for the sake of simplicity.

Finally, I normalized the dataset becuase it had values from different scales.

-------------

**Second: Creating The Model** - `main.py`

- This part was easier and took much less time comparing to the first part.
- I started of with a simple Logestic Regression Classifier but it did not get me anywhere, so I opted for a Neural Net using `MLPClassifier`
- Arbitrarily, I chose the default model with an architecture of four layers, three of them having 5 units and the last one had 2. Later on I tuned this but got no where better so I got back to my initial settings.
- My model was overfitting the dataset due to the large number of features compared to the amount of given data, so I tried to select the features, but having over 50 feature (after extracting dummies) with no clue what each feature represent, I randomly chose features and by try and error I selected the set with the highest accuracy - ~86%
	- *All random chosen features were saved in a text file named `accuracies_for_different_set_of_features.txt`*
- I then started tuning the regularizaton parameter and got the best results at alpha = 0.1.
- Finally I tried out other `solvers` and `activation` functions but got no where better than I was.