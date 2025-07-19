import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split

# Step 1: Load Data
df = pd.read_csv('data.csv')
print("Original Data:\n", df)

# Step 2: Clean Data (Missing values)
df['age'].fillna(df['age'].mean(), inplace=True)
df['salary'].fillna(df['salary'].mean(), inplace=True)

# Step 3: Encode Categorical Columns
le_gender = LabelEncoder()
df['gender'] = le_gender.fit_transform(df['gender'])  # Male=1, Female=0

df = pd.get_dummies(df, columns=['city'])  # One-hot encode 'city'

# Step 4: Drop non-numeric column 'name'
df = df.drop('name', axis=1)

# Step 5: Normalize numerical columns
scaler = MinMaxScaler()
df[['age', 'salary']] = scaler.fit_transform(df[['age', 'salary']])

# Step 6: Split into train/test
X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Print shapes for verification
print("\nProcessed Data:\n", df)
print("\nX_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
