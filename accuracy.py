import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# load the iris dataset as a pandas dataframe
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
df = pd.read_csv(url, names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species'])
print(df)

# create a feature matrix and target vector
X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y = df['species']

# split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# create a pipeline object
pipeline = Pipeline(steps=[
    ('scaler', StandardScaler()),
    ('model', LogisticRegression())
])

# fit the pipeline to the training data
pipeline.fit(X_train, y_train)

# make predictions on the test data
y_pred = pipeline.predict(X_test)

# evaluate the model's accuracy
acc = accuracy_score(y_test, y_pred)
print(f'Test accuracy: {acc:.2f}')

# try a different model
pipeline = Pipeline(steps=[
    ('scaler', StandardScaler()),
    ('model', RandomForestClassifier())
])

# fit the pipeline to the training data
pipeline.fit(X_train, y_train)

# make predictions on the test data
y_pred = pipeline.predict(X_test)

# evaluate the model's accuracy
acc = accuracy_score(y_test, y_pred)
print(f'Test accuracy: {acc:.2f}')
