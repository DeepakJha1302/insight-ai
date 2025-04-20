from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import pandas as pd

def analyze_data(df, analysis_type):
    y = df.iloc[:, -1]
    X = df.iloc[:, :-1]

    if analysis_type == "descriptive":
        model = LinearRegression()
    elif analysis_type == "prescriptive":
        model = DecisionTreeClassifier()
    else:
        model = KMeans(n_clusters=3)

    X_train, X_test, y_train, y_test = train_test_split(X, y)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    return pd.DataFrame({"Actual": y_test, "Predicted": predictions}), str(model)
