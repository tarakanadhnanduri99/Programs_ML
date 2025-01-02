import pandas as pd
import numpy as np
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score
import pickle

# Load the data
data: DataFrame = pd.read_csv("C:\\Users\hp\Downloads\Bengaluru_House_Data.csv")

# Data exploration and cleaning
print(data.head())
print(data.shape)
print(data.info())

for column in data.columns:
    print(data[column].value_counts())
    print("*" * 20)

print(data.isna().sum())

# Dropping unnecessary columns
data.drop(columns=['area_type', 'availability', 'society', 'balcony'], inplace=True)

print(data.describe())

# Fill missing values
data['location'] = data['location'].fillna('Sarjapur Road')
data['size'] = data['size'].fillna('2 BHK')
data['bath'] = data['bath'].fillna(data['bath'].median())

# Convert BHK to an integer
data['bhk'] = data['size'].str.split().str.get(0).astype(int)

# Handle total_sqft, convert ranges to average and remove anomalies
def convertRange(x):
    temp = x.split('-')
    if len(temp) == 2:
        return (float(temp[0]) + float(temp[1])) / 2
    try:
        return float(x)
    except:
        return None

data['total_sqft'] = data['total_sqft'].apply(convertRange)

# Price per square foot calculation
data['price_per_sqft'] = data['price'] * 1000000 / data['total_sqft']

# Clean location names by stripping whitespaces
data['location'] = data['location'].apply(lambda x: x.strip())

# Handling rare locations
location_count = data['location'].value_counts()
location_count_less_10 = location_count[location_count <= 10]
data['location'] = data['location'].apply(lambda x: 'other' if x in location_count_less_10 else x)

# Remove outliers based on total_sqft per BHK
data = data[((data['total_sqft'] / data['bhk']) >= 300)]

# Remove outliers based on price per sqft within each location
def remove_outliers_sqft(df):
    df_output = pd.DataFrame()
    for key, subdf in df.groupby('location'):
        m = np.mean(subdf.price_per_sqft)
        st = np.std(subdf.price_per_sqft)
        gen_props = subdf[(subdf.price_per_sqft > (m - st)) & (subdf.price_per_sqft <= (m + st))]
        df_output = pd.concat([df_output, gen_props], ignore_index=True)
    return df_output

data = remove_outliers_sqft(data)

# Remove BHK outliers
def bhk_outlier_remover(df):
    exclude_indices = np.array([])
    for location, location_df in df.groupby('location'):
        bhk_stats = {}
        for bhk, bhk_df in location_df.groupby('bhk'):
            bhk_stats[bhk] = {
                'mean': np.mean(bhk_df.price_per_sqft),
                'std': np.std(bhk_df.price_per_sqft),
                'count': bhk_df.shape[0]
            }
        for bhk, bhk_df in location_df.groupby('bhk'):
            stats = bhk_stats.get(bhk - 1)
            if stats and stats['count'] > 5:
                exclude_indices = np.append(exclude_indices, bhk_df[bhk_df.price_per_sqft < stats['mean']].index.values)
    return df.drop(exclude_indices, axis='index')

data = bhk_outlier_remover(data)

# Dropping unnecessary columns
data.drop(columns=['size', 'price_per_sqft'], inplace=True)

# Splitting the data into features and target
X = data.drop(columns=['price'])
y = data['price']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

print(X_train.shape)
print(X_test.shape)

# Applying Linear Regression
column_trans = make_column_transformer(
    (OneHotEncoder(sparse_output=False), ['location']),
    remainder='passthrough'
)

scalar = StandardScaler()
lr = LinearRegression()

pipe = make_pipeline(column_trans, scalar, lr)
pipe.fit(X_train, y_train)

# Predictions and evaluation
y_pred_lr = pipe.predict(X_test)
print("Linear Regression R2 Score:", r2_score(y_test, y_pred_lr))

# Applying Lasso Regression
lasso = Lasso()
pipe = make_pipeline(column_trans, scalar, lasso)
pipe.fit(X_train, y_train)
y_pred_lasso = pipe.predict(X_test)
print("Lasso Regression R2 Score:", r2_score(y_test, y_pred_lasso))

# Applying Ridge Regression
ridge = Ridge()
pipe = make_pipeline(column_trans, scalar, ridge)
pipe.fit(X_train, y_train)
y_pred_ridge = pipe.predict(X_test)
print("Ridge Regression R2 Score:", r2_score(y_test, y_pred_ridge))

# Save the best model (Ridge in this case)
pickle.dump(pipe, open('RidgeModel.pkl', 'wb'))
