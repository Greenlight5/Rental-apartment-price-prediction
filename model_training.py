from assets_data_prep import prepare_data
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from assets_data_prep import prepare_data
from sklearn.linear_model import ElasticNet
import pickle
import joblib
import warnings

warnings.simplefilter(action='ignore', category=RuntimeWarning)

train_df = pd.read_csv("train.csv")

#we split the data manually befor applying the prepare data in order to  use it as a test file (that's why it doesnt appears here)
df = prepare_data(train_df, "train")
X = df.drop("price", axis=1)
y = df['price']

#split the data to train and test before changing thinks: 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_cols = X.select_dtypes(include=["object", "bool"]).columns.tolist()

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numeric_cols),
    ("cat",TargetEncoder(), categorical_cols)])

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", ElasticNet(max_iter=10000))])


param_grid = {
    "model__alpha": [0.01, 0.1, 1.0, 10.0],
    "model__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9]}

en_model = GridSearchCV(pipeline, param_grid, cv=10, scoring="neg_mean_squared_error")
en_model.fit(X_train, y_train)

y_pred = en_model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)


# Save the trained pipeline to a pickle file
pickle.dump(en_model, open('trained_model.pkl', 'wb'))

# Assuming you have already fitted your preprocessor on your training data
with open('preprocessor.pkl', 'wb') as file:
    joblib.dump(preprocessor, file)