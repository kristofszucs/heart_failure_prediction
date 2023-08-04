import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import make_column_selector as selector
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
import joblib

df = pd.read_csv("./data.csv")
df['FastingBS'] = df['FastingBS'].replace({0: 'No', 1: 'Yes'})
X = df.drop(['HeartDisease'], axis=1)
y =  df['HeartDisease']

numerical_columns_selector = selector(dtype_include = ["int64","float64"])
categorical_columns_selector = selector(dtype_exclude = ["int64","float64"])

numerical_columns = numerical_columns_selector(X)
categorical_columns = categorical_columns_selector(X)

categorical_preprocessor = OneHotEncoder(handle_unknown="ignore")
numerical_preprocessor = StandardScaler()

preprocessor = ColumnTransformer(
    [
        ("one-hot-encoder", categorical_preprocessor, categorical_columns),
        ("standard_scaler", numerical_preprocessor, numerical_columns),
    ]
)
model_rf = RandomForestClassifier()

pipe_rf = make_pipeline(preprocessor, model_rf)

param_rf = {'randomforestclassifier__n_estimators': [100, 200, 500],
    'randomforestclassifier__max_features': ['auto', 'sqrt', 'log2'],
    'randomforestclassifier__max_depth' : [None,4,6,8],
    'randomforestclassifier__criterion' :['gini', 'entropy']
}

grid_rf = GridSearchCV(pipe_rf, param_rf, cv = 5)
grid_rf.fit(X, y)

#print(grid_rf.predict(X_test)[:5])
#print(grid_rf.score(X_test, y_test))
#print("Best hyperparameters : "+ str(grid_rf.best_params_))
joblib.dump(grid_rf, 'random_forest_model.joblib')
