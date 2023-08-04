import pandas as pd
import joblib
import warnings
warnings.filterwarnings('ignore')

grid_rf = joblib.load('./random_forest_model.joblib')
df_validation = pd.read_csv("./model_validation_data.csv")

df_validation['FastingBS'] = df_validation['FastingBS'].replace({0: 'No', 1: 'Yes'})

X_valid = df_validation.drop('HeartDisease', axis=1)
y_valid = df_validation['HeartDisease']

# best_pipeline = grid_rf.best_estimator_
# df_validation_preprocessed = best_pipeline.named_steps['columntransformer'].transform(df_validation)

predictions = grid_rf.predict(X_valid)
predictions_proba = grid_rf.predict_proba(X_valid)

print(predictions)
#print(predictions_proba)

#from sklearn.metrics import confusion_matrix
#print(confusion_matrix(y_valid, predictions))

#from sklearn.metrics import classification_report
#print(classification_report(y_valid, predictions))