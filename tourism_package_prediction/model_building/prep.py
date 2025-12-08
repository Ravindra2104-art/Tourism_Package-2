# for data manipulation
import pandas as pd
import sklearn
# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
# for converting text data in to numerical representation
from sklearn.preprocessing import LabelEncoder
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi
from datasets import Dataset, load_dataset # Import Dataset and load_dataset class

# Load data from HuggingFace
HF_TOKEN = os.getenv("HF_TOKEN")
login(HF_TOKEN)
try:
    dataset = load_dataset("RavindraDubey12/Tourism-Package-Prediction", split="train")
    df = dataset.to_pandas()
    if "__index_level_0__" in df.columns:
       df = df.drop(columns="__index_level_0__")
    print(f"Dataset loaded from HuggingFace: {len(df)} rows")
except:
    df = pd.read_csv("tourism_package_prediction/data/tourism.csv")
    print(f"Dataset loaded locally: {len(df)} rows")

# Data cleaning and preprocessing
print("Starting data cleaning...")

# Remove unnecessary columns
if 'Unnamed: 0' in df.columns:
    df = df.drop('Unnamed: 0', axis=1)

# Handle missing values
print("Missing values before cleaning:")
print(df.isnull().sum()[df.isnull().sum() > 0])

# Fill missing values
numerical_cols = ['Age', 'DurationOfPitch', 'NumberOfFollowups', 'PreferredPropertyStar',
                 'NumberOfTrips', 'PitchSatisfactionScore', 'NumberOfChildrenVisiting', 'MonthlyIncome']

for col in numerical_cols:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].median())

categorical_cols = ['TypeofContact', 'Occupation', 'Gender', 'MaritalStatus',
                   'ProductPitched', 'Designation']

for col in categorical_cols:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown')

# Fix data inconsistencies
df['Gender'] = df['Gender'].replace('Fe Male', 'Female')

print("Missing values after cleaning:")
print(df.isnull().sum()[df.isnull().sum() > 0])

# Feature engineering
print("Feature engineering...")

# Create income categories
df['IncomeCategory'] = pd.cut(df['MonthlyIncome'],
                             bins=[0, 15000, 25000, 35000, float('inf')],
                             labels=[0, 1, 2, 3])  # Use numeric labels

# Create age groups
df['AgeGroup'] = pd.cut(df['Age'],
                       bins=[0, 25, 35, 45, 55, float('inf')],
                       labels=[0, 1, 2, 3, 4])  # Use numeric labels

# Encode categorical variables
label_encoders = {}
categorical_columns = df.select_dtypes(include=['object']).columns

for col in categorical_columns:
    if col != 'CustomerID':
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

print(f"Data preprocessing completed! Final shape: {df.shape}")

# Split data
print("Splitting data...")
X = df.drop(['CustomerID', 'ProdTaken'], axis=1)
y = df['ProdTaken']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Create train and test dataframes
train_df = pd.concat([X_train, y_train], axis=1)
test_df = pd.concat([X_test, y_test], axis=1)

# Add CustomerID
train_df.insert(0, 'CustomerID', range(300000, 300000 + len(train_df)))
test_df.insert(0, 'CustomerID', range(400000, 400000 + len(test_df)))

# Save locally
train_df.to_csv("tourism_package_prediction/data/train_data.csv", index=False)
test_df.to_csv("tourism_package_prediction/data/test_data.csv", index=False)

print(f"Data split completed!")
print(f"Training set: {len(train_df)} samples")
print(f"Test set: {len(test_df)} samples")

# Upload train dataset
train_dataset = Dataset.from_pandas(train_df,preserve_index=False)
train_dataset.push_to_hub(
    "RavindraDubey12/Tourism-Package-Prediction-train",
    private=False,
    token=HF_TOKEN
)

# Upload test dataset
test_dataset = Dataset.from_pandas(test_df,preserve_index=False)
test_dataset.push_to_hub(
    "RavindraDubey12/Tourism-Package-Prediction-test",
    private=False,
    token=HF_TOKEN
)
print("Processed datasets uploaded to HuggingFace!")
