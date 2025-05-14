# 📌 Phase 1: Setup and Data Loading
!pip install seaborn xgboost plotly
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, classification_report, confusion_matrix
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import xgboost as xgb
from google.colab import files
uploaded = files.upload()
df = pd.read_csv('city_day.csv')  # Replace with your actual file name
df.head()

df.ffill(inplace=True)
df.bfill(inplace=True)

print(df.columns)

location_column_name = 'City'
df.drop_duplicates(subset=['Date', location_column_name], inplace=True)
df['Date'] = pd.to_datetime(df['Date'])

label_encoder = LabelEncoder()
try:
    df['AQI_Category_Encoded'] = label_encoder.fit_transform(df['AQI_Bucket'])  # Use correct label column
except KeyError:
    print("Column 'AQI_Bucket' not found in the DataFrame. Please check the column name.")

df_with_dummies = pd.get_dummies(df, columns=[location_column_name])

# Drop all non-numeric or unwanted columns before scaling
columns_to_drop = ['AQI', 'AQI_Bucket', 'AQI_Category_Encoded', 'Date']
df_for_scaling = df_with_dummies.drop(columns=columns_to_drop, axis=1, errors='ignore')

scaled_features = MinMaxScaler().fit_transform(df_for_scaling)

X = pd.DataFrame(scaled_features, columns=df_for_scaling.columns)


# 📌 Phase 3: Exploratory Data Analysis (EDA)
plt.figure(figsize=(10, 6))
# Select only numerical features for correlation
numerical_df = df.select_dtypes(include=np.number)  # Select numerical columns
sns.heatmap(numerical_df.corr(), annot=True)  # Calculate correlation on numerical data
plt.title('Correlation Heatmap')
plt.show()
plt.figure(figsize=(12, 6))
df.groupby('Date')['AQI'].mean().plot()  # Changed 'timestamp' to 'Date'
plt.title('AQI Trend Over Time')
plt.show()

# 📌 Phase 4: Feature Engineering
df['day'] = df['Date'].dt.day  # Changed 'timestamp' to 'Date'
df['month'] = df['Date'].dt.month  # Changed 'timestamp' to 'Date'
df['year'] = df['Date'].dt.year  # Changed 'timestamp' to 'Date'
df['season'] = df['month'] % 12 // 3 + 1

# 📌 Phase 5: Modeling - Regression
# --- Ensure X_reg has the correct columns ---
# Check columns in df: print(df.columns)
# Based on the available columns, select the features for regression
# Adjust the list of columns to drop if 'AQI_Category' is not present
# Assuming your target is 'AQI' and 'AQI_Category' is not present
# If 'AQI_Category' is present, keep it in the list of columns to drop

# Exclude the 'Date' column from the features
# --- Drop 'AQI_Category' along with 'AQI' and 'Date' ---
# X_reg = df.drop(columns=['AQI', 'Date', 'AQI_Category'])  # Dropped 'Date' and 'AQI_Category' columns
# Instead of dropping 'AQI_Category', drop unnecessary columns based on your data
# For example, if you only need numerical features:
#X_reg = df.drop(columns=['AQI', 'Date', 'AQI_Category', 'AQI_Category_Encoded'])  # Dropped 'Date' and 'City' columns. Removed 'AQI_Category'
# Or, list out the specific columns you want to keep:
# Modified to only drop existing columns:
# If 'AQI_Category_Encoded' is present in the DataFrame

# --- Explicitly select numerical features ---
X_reg = df.select_dtypes(include=np.number)  # Select only numerical columns

# --- Drop target and unnecessary columns ---
X_reg = X_reg.drop(columns=['AQI', 'AQI_Category_Encoded', 'day', 'month', 'year', 'season'], errors='ignore') # Dropped 'AQI', 'Date', and 'AQI_Category', 'AQI_Category_Encoded' if exists



#X_reg = df[['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene', 'day', 'month', 'year', 'season']]  # Selected specific columns

y_reg = df['AQI']
X_train, X_test, y_train, y_test = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
model_rf = RandomForestRegressor()
model_rf.fit(X_train, y_train)
y_pred_rf = model_rf.predict(X_test)
print('MAE:', mean_absolute_error(y_test, y_pred_rf))
print('RMSE:', np.sqrt(mean_squared_error(y_test, y_pred_rf)))
print('R2 Score:', r2_score(y_test, y_pred_rf))

from IPython import get_ipython
from IPython.display import display
!pip install seaborn xgboost plotly
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, classification_report, confusion_matrix
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import xgboost as xgb
from google.colab import files

uploaded = files.upload()
df = pd.read_csv('city_day.csv')
df.head()

df.fillna(method='ffill', inplace=True)
df.fillna(method='bfill', inplace=True)
print(df.columns)
location_column_name = 'City'
df.drop_duplicates(subset=['Date', location_column_name], inplace=True)
df['Date'] = pd.to_datetime(df['Date'])

label_encoder = LabelEncoder()
try:
    df['AQI_Category_Encoded'] = label_encoder.fit_transform(df['AQI_Bucket'])  # Correct column name
except KeyError:
    print("Column 'AQI_Bucket' not found in the DataFrame. Please check the column name.")


df_with_dummies = pd.get_dummies(df, columns=[location_column_name], dtype=float)  # Specify dtype=float for dummies
scaler = MinMaxScaler()
columns_to_drop = ['AQI', 'Date', 'AQI_Bucket', 'AQI_Category_Encoded']

# errors='ignore' handles the case if a column doesn't exist

# Check if the columns are present before dropping
if 'AQI_Category' in df_with_dummies.columns:
    columns_to_drop.append('AQI_Category')  # Ensure AQI_Category is dropped if it exists
if 'AQI_Category_Encoded' in df_with_dummies.columns:
    columns_to_drop.append('AQI_Category_Encoded')  # Ensure AQI_Category_Encoded is dropped if it exists

scaled_features = scaler.fit_transform(df_with_dummies.drop(columns=columns_to_drop, axis=1, errors='ignore'))
X = pd.DataFrame(scaled_features)

plt.figure(figsize=(10, 6))
numerical_df = df.select_dtypes(include=np.number)
sns.heatmap(numerical_df.corr(), annot=True)
plt.title('Correlation Heatmap')
plt.show()

plt.figure(figsize=(12, 6))
df.groupby('Date')['AQI'].mean().plot()
plt.title('AQI Trend Over Time')
plt.show()

df['day'] = df['Date'].dt.day
df['month'] = df['Date'].dt.month
df['year'] = df['Date'].dt.year
df['season'] = df['month'] % 12 // 3 + 1

X_reg = df.select_dtypes(include=np.number)
X_reg = X_reg.drop(columns=['AQI', 'AQI_Category_Encoded', 'day', 'month', 'year', 'season'], errors='ignore')
y_reg = df['AQI']

X_train, X_test, y_train, y_test = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
model_rf = RandomForestRegressor()
model_rf.fit(X_train, y_train)
y_pred_rf = model_rf.predict(X_test)

print('MAE:', mean_absolute_error(y_test, y_pred_rf))
print('RMSE:', np.sqrt(mean_squared_error(y_test, y_pred_rf)))
print('R2 Score:', r2_score(y_test, y_pred_rf))

X_c = df_with_dummies.drop(columns=['AQI_Category', 'AQI', 'AQI_Category_Encoded', 'Date'], errors='ignore')
y_c = df['AQI_Bucket']


X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_c, y_c, test_size=0.2, random_state=42)
clf = DecisionTreeClassifier()
clf.fit(X_train_c, y_train_c)
y_pred_c = clf.predict(X_test_c)

print(classification_report(y_test_c, y_pred_c))
sns.heatmap(confusion_matrix(y_test_c, y_pred_c), annot=True)
plt.title('Confusion Matrix')
plt.show()

try:
    feat_imp = pd.Series(model_rf.feature_importances_, index=X_train.columns)
    feat_imp.nlargest(10).plot(kind='barh')
    plt.title('Top 10 Feature Importances')
    plt.show()
except NameError:
    print("Error: model_rf is not defined. Please ensure the model is trained in a previous cell.")


# 📌 Phase 7: Feature Importance
# Assuming model_rf is defined and trained in a previous cell
# If not, make sure to train the model before this cell
try:
    feat_imp = pd.Series(model_rf.feature_importances_, index=X_train.columns)
    feat_imp.nlargest(10).plot(kind='barh')
    plt.title('Top 10 Feature Importances')
    plt.show()
except NameError:
    print("Error: model_rf is not defined. Please ensure the model is trained in a previous cell.")