import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from category_encoders import TargetEncoder
import shap


def load_data(file_path):
  """
  Load data from a CSV file into a pandas DataFrame.
  Args:
    file_path (str): The path to the CSV file.
  Returns:
    df (pandas.DataFrame): The loaded DataFrame.
  """
  try:
    df = pd.read_csv(file_path)
    print("Data loaded successfully!")
  except FileNotFoundError:
    print("File not found. Please check the file path.")
  except Exception as e:
    print(f"An error occurred: {e}")
  
  return df


def data_exploration(df):
  """
  Generates a comprehensive initial exploration report for a given DataFrame.
  Args:
    df (pandas.DataFrame): The input DataFrame for exploration.
  Returns:
    columns_with_missing_values_df (pandas.DataFrame): A DataFrame containing columns with missing values.
    high_missing_columns (list): A list of column names with missing values exceeding a threshold.
  """
  print("#" * 50)
  print("#            Data Exploration Report:            #")
  print("#" * 50)

  # Shape
  num_rows, num_cols = df.shape
  print(f"\n1. Dataset Shape: {num_rows} rows, {num_cols} columns")

  # Data Types Summary
  print("\n2. Data Types Summary:")
  dtypes_summary = df.dtypes.value_counts()
  for dtype, count in dtypes_summary.items():
    print(f"    - {dtype}: {count} columns")

  # Missing Value Analysis (The Critical Step)
  print("\n3. Missing Value Analysis:")
  missing_values = df.isnull().sum()
  missing_percentage = (missing_values / num_rows) * 100

  # Create a summary DataFrame for missing values
  missing_summary = pd.DataFrame({
      'Missing Values': missing_values,
      'Missing Percentage': missing_percentage
  }).sort_values(by='Missing Percentage', ascending=False)

  # Display the summary DataFrame
  # print(missing_summary)

  # Filter and display columns with missing values only
  columns_with_missing_values_df = missing_summary[missing_summary['Missing Values'] > 0]

  if not columns_with_missing_values_df.empty:
    print(f"\n   Total columns with missing values: {len(columns_with_missing_values_df)}")
    print("\n   Columns with Missing Values (Top 10):")
    print(columns_with_missing_values_df.head(10))

    # Flag high missing columns
    high_missing_columns = columns_with_missing_values_df[columns_with_missing_values_df['Missing Percentage'] > 40].index.to_list()
    if high_missing_columns:
      print(f"\n   ({len(high_missing_columns)}) columns with High Missing Values (>40%): {high_missing_columns}")
  else:
    print("   No columns have missing values.")

  # Target Variable Preview (Crucial for our business problem)
  if 'SalePrice' in df.columns:
    print("\n4. Target Variable ('SalePrice') Preview:")
    print(f"   Minimum Sale Price: ${df['SalePrice'].min():,}")
    print(f"   Maximum Sale Price: ${df['SalePrice'].max():,}")
    print(f"   Avarage Sale Price: ${df['SalePrice'].mean():,.0f}")
  
  # Sampling for Initial View
  print("\n5. Sampling (Stratified by a Key Categorical Column) for Initial View:")
  # Using 'SaleCondition' as an example to ensure sample diversity
  if 'SaleCondition' in df.columns:
    sample_df = df.groupby('SaleCondition', group_keys=False).apply(lambda x: x.sample(min(len(x), 2)), include_groups=False)
    sample_df['SaleCondition'] = df['SaleCondition']
    print(sample_df[['SaleCondition', 'SalePrice', 'GrLivArea', 'Neighborhood']])
  else:
    # First 10 columns of a random sample
    print(df.sample(10, random_state=42).iloc[:, :10])

  return columns_with_missing_values_df, high_missing_columns if 'high_missing_columns' in locals() else []


def imputation_strategy(df, high_missing_columns):
  """
  Create imputation strategies for different columns based on their data type.
  Args:
    df (pandas.DataFrame): The input DataFrame.
    high_missing_columns (list): A list of column names with missing values exceeding a threshold.
  Returns:
    columns_to_drop (list): A list of columns to be dropped.
    categorical_imputation (dict): A dictionary of imputation strategies for categorical columns.
    continuous_imputation (dict): A dictionary of imputation strategies for continuous columns.
  """

  print("#" * 50)
  print("#            Data Cleaining Report:            #")
  print("#" * 50)
  print()

  # Columns with meaningful "None" categories
  columns_to_protect = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu']

  # Columns to drop entirely
  columns_to_drop = [col for col in high_missing_columns if col not in columns_to_protect]

  ordinal_columns = [
      'LotShape', 'Utilities', 'LandSlope', 'OverallQual', 'OverallCond',
      'ExterQual', 'ExterCond', 'BsmtQual','BsmtCond', 'BsmtExposure', 'BsmtFinType1',
      'BsmtFinType2', 'HeatingQC', 'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu',
      'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 'Fence'
  ]

  nominal_columns = [
      'MSZoning', 'Street', 'Alley', 'LandContour',
      'LotConfig', 'Neighborhood', 'Condition1', 'Condition2',
      'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',
      'Exterior2nd', 'MasVnrType', 'Foundation', 'Heating', 'CentralAir',
      'GarageType', 'MiscFeature', 'SaleType', 'SaleCondition'
  ]

  numerical_columns = [
      'LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2',
      'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF',
      'GrLivArea', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF',
      'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal'
  ]

  # Categorical data Strategy dictionary
  all_categorical = ordinal_columns + nominal_columns
  categorical_imputation = {col: 'None' for col in all_categorical}

  # Continuous: 0 or median imputation
  continuous_imputation = {}
  for col in numerical_columns:
    if 'Area' in col or 'SF' in col or 'Porch' in col:
      continuous_imputation[col] = 0
    else:
      neighborhood_median = df.groupby('Neighborhood')[col].transform('median')

      global_median = df[col].median()

      continuous_imputation[col] = neighborhood_median.fillna(global_median).fillna(0)

  return columns_to_drop, categorical_imputation, continuous_imputation


def clean_data(df, high_missing_columns):
  """
  Clean the input DataFrame based on the provided imputation strategies.
  Args:
    df (pandas.DataFrame): The input DataFrame.
    high_missing_columns (list): A list of column names with missing values exceeding a threshold.
  Returns:
    df_cleaned (pandas.DataFrame): The cleaned DataFrame.
    log (dict): A dictionary containing the cleaning process log.
  """
  df_cleaned = df.copy()
  log = {}

  # Get imputation strategies
  columns_to_drop, categorical_imputation, continuous_imputation = imputation_strategy(df_cleaned, high_missing_columns)

  # Drop columns
  df_cleaned.drop(columns=columns_to_drop, inplace=True)
  log['columns_dropped'] = columns_to_drop

  # Before imputation, create missing value indicators
  missing_indicators = {}
  indicator_columns = [
      'GarageYrBlt', 'MasVnrArea', 'LotFrontage', 'BsmtFinSF1', 'BsmtFinSF2',
      'BsmtUnfSF', 'TotalBsmtSF'
  ]
  for col in indicator_columns:
    indicator_name = f"{col}_missing"
    if col in df_cleaned.columns and df_cleaned[col].isnull().any():
      df_cleaned[indicator_name] = df_cleaned[col].isnull().astype(int)
      missing_indicators[col] = indicator_name

  log['missing_indicators_created'] = list(missing_indicators.values())

  # Categorical imputation
  categorical_imputed = {}
  for col, value in categorical_imputation.items():
    if col in df_cleaned.columns and df_cleaned[col].isnull().any():
      df_cleaned[col] = df_cleaned[col].fillna(value)
      categorical_imputed[col] = value

  log['categorical_imputed'] = list(categorical_imputation.keys())

  # Continuous imputation
  continuous_imputed = {}
  for col, value in continuous_imputation.items():
    if col in df_cleaned.columns and df_cleaned[col].isnull().any():
      df_cleaned[col] = df_cleaned[col].fillna(value)
      continuous_imputed[col] = value

  log['continuous_imputed'] = list(continuous_imputation.keys())

  discrete_columns = [
      'BsmtFullBath', 'BsmtHalfBath',
      'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr',
      'TotRmsAbvGrd', 'Fireplaces', 'GarageCars'
  ]

  temporal_columns = [
      'YearBuilt', 'YearRemodAdd', 'GarageYrBlt', 'MoSold', 'YrSold'
  ]

  # Discrete imputation
  discrete_imputed = {}
  for col in discrete_columns:
    if col in df_cleaned.columns and df_cleaned[col].isnull().any():
      df_cleaned[col] = df_cleaned[col].fillna(0)
      discrete_imputed[col] = 0

  log['discrete_imputed'] = list(discrete_imputed.keys())

  # Temporal imputation
  temporal_imputed = {}
  for col in temporal_columns:
    if col in df_cleaned.columns and df_cleaned[col].isnull().any():
      if col == 'GarageYrBlt' or col == 'YearRemodAdd':
        df_cleaned[col] = df_cleaned[col].fillna(df_cleaned['YearBuilt'])
        temporal_imputed[col] = 'YearBuilt'
      elif col == 'MoSold' or col == 'YrSold':
        mode_value = df_cleaned[col].mode()[0]
        df_cleaned[col] = df_cleaned[col].fillna(mode_value)
        temporal_imputed[col] = 'Mode'
      else:
        df_cleaned[col] = df_cleaned.groupby('Neighborhood')[col].transform(lambda x: x.fillna(x.median()))
        temporal_imputed[col] = 'Median'

  log['temporal_imputed'] = list(temporal_imputed.keys())

  # Final verification of missing values
  remaining_missing = df_cleaned.isnull().sum().sum()
  log['remaining_missing'] = remaining_missing
  log['cleaning_completed'] = remaining_missing == 0

  return df_cleaned, log


def engineer_features(df):
  """
  Create new features based on existing ones.
  Args:
    df (pandas.DataFrame): The input DataFrame.
  Returns:
    df_engineered (pandas.DataFrame): The DataFrame with engineered features.
  """

  print("#" * 50)
  print("#            Feature Engineering Report:            #")
  print("#" * 50)
  print()

  df_engineered = df.copy()

  # Aggregate features such total bathrooms and total porchs
  df_engineered['TotalBathroom'] = df_engineered['FullBath'] + 0.5 * df_engineered['HalfBath'] + df_engineered['BsmtFullBath'] + 0.5 * df_engineered['BsmtHalfBath']

  porch_cols = ['OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch']
  df_engineered['TotalPorch'] = df_engineered[porch_cols].sum(axis=1)

  # Total SF
  sf_cols = ['TotalBsmtSF', '1stFlrSF', '2ndFlrSF']
  df_engineered['TotalSF'] = df_engineered[sf_cols].sum(axis=1)

  # House age at sale
  df_engineered['HouseAge'] = df_engineered['YrSold'] - df_engineered['YearBuilt']

  # Remodeled age at sale
  df_engineered['RemodAge'] = df_engineered['YrSold'] - df_engineered['YearRemodAdd']
  df_engineered['RemodAge'] = df_engineered['RemodAge'].apply(lambda x: max(0, x))

  # Quality - Area SF interaction
  df_engineered['QualArea'] = df_engineered['OverallQual'] * df_engineered['TotalSF']

  return df_engineered


def analyze_nominal_cardinality(df, nominal_columns):
  """
  Analyze the cardinality of nominal columns in the DataFrame.
  Args:
    df (pandas.DataFrame): The input DataFrame.
    nominal_columns (list): A list of nominal columns to analyze.
  Returns:
    cardinalities (dict): A dictionary containing column names and their cardinalities.
    low_cardinality (list): A list of columns with low cardinality.
    high_cardinality (list): A list of columns with high cardinality.
  """
  cardinality_info = {}
  for col in nominal_columns:
    if col in df.columns:
      cardinality_info[col] = {
          'number_of_categories': df[col].nunique(),
          'categories': sorted(df[col].unique().tolist())
      }

  # Categorize columns based on cardinality
  low_cardinality = [col for col, info in cardinality_info.items() if info['number_of_categories'] <= 10]
  high_cardinality = [col for col, info in cardinality_info.items() if info['number_of_categories'] > 10]

  print(f"Number of columns with low cardinality (<= 10 categories): {len(low_cardinality)}")
  print(f"Number of columns with high cardinality (> 10 categories): {len(high_cardinality)}")

  return cardinality_info, low_cardinality, high_cardinality


def create_encoding(df, ordinal_mapping, low_cardinality_nominal_columns, high_cardinality_nominal_columns):
  """
  Create encoding strategies for different columns based on their data type.
  Args:
    df (pandas.DataFrame): The input DataFrame.
    ordinal_mapping (dict): A dictionary mapping ordinal categories to numerical values.
    low_cardinality (list): A list of columns with low cardinality.
    high_cardinality (list): A list of columns with high cardinality.
  Returns:
    ordinal_encoder (OrdinalEncoder): An ordinal encoder for ordinal columns.
    low_cardinality_encoder (OneHotEncoder): A one-hot encoder for low cardinality columns.
    high_cardinality_encoder (TargetEncoder): A target encoder for high cardinality columns.
  """
  df_encoded = df.copy()
  encoders = {}
  target_column = 'SalePrice'

  # Ordinal encoding
  print("\nEncoding ordinal columns...")
  for col, ratings in ordinal_mapping.items():
    if col in df_encoded.columns:
      # Ensure the column is of object type before mapping, to handle 'None' string consistently
      df_encoded[col] = df_encoded[col].astype('object')
      df_encoded[col] = df_encoded[col].map(ratings)
      # Fill any NaNs that might arise from mapping new/unseen categories with the 'None' value (0)
      df_encoded[col].fillna(0, inplace=True)

  print("\nOrdinal encoding completed.")

  # One-hot encoding Low Cardinality Nominal columns
  print(f"\nOne-hot encoding {len(low_cardinality_nominal_columns)} low-cardinality columns...")
  for col in low_cardinality_nominal_columns:
    if col in df_encoded.columns:
      # Ensure the column is of object type before one-hot encoding
      df_encoded[col] = df_encoded[col].astype('object')
      # Removed handle_unknown='ignore' as it's primarily for transform, not fit_transform
      encoder = OneHotEncoder(sparse_output=False, drop='first')

      # Fit and Transform the column
      transformed_column = encoder.fit_transform(df_encoded[[col]])

      # Use get_feature_names_out for robust column naming and handling 'drop=first'
      ohe_column_names = encoder.get_feature_names_out([col])

      # Create DataFrame for the new one-hot encoded column, ensuring index alignment
      if len(ohe_column_names) > 0:
        ohe_df = pd.DataFrame(transformed_column, index=df_encoded.index, columns=ohe_column_names)

        # Concatenate the new one-hot encoded columns to the original DataFrame
        df_encoded = pd.concat([df_encoded, ohe_df], axis=1)

      # Drop original column after encoding, or if it had only one category and was skipped by OHE
      df_encoded.drop(columns=[col], inplace=True)

      # Store the encoder for future use
      encoders[f"onehot_{col}"] = encoder

  print("\nOne-hot encoding completed.")

  # Target encode high-cardinality nominal columns
  print(f"\nTarget encoding {len(high_cardinality_nominal_columns)} high-cardinality columns...")
  for col in high_cardinality_nominal_columns:
    if col in df_encoded.columns:
      # Ensure the column is of object type before target encoding
      df_encoded[col] = df_encoded[col].astype('object')
      # Initialize the encoder by specifying the column to encode
      encoder = TargetEncoder(cols=[col], smoothing=10, min_samples_leaf=5)

      # Fit and Transform the column
      # TargetEncoder expects X to be a 2D array and y to be a 1D array
      encoded_column = encoder.fit_transform(df_encoded[[col]], df_encoded[target_column])
      df_encoded[f"{col}_encoded"] = encoded_column

      # Drop original column after encoding
      df_encoded.drop(columns=[col], inplace=True)

      # Store the encoder for future use
      encoders[f"target_{col}"] = encoder

  print("\nTarget encoding completed.")

  return df_encoded, encoders


def get_raw_training_stats(df, target_column):
    """
    Calculate stats (median/mode) on the raw, un-encoded data.
    This allows us to fill in missing user inputs with defaults.
    """
    # Drop target if present
    X = df.drop(columns=[target_column], errors='ignore')
    
    stats = {
        'medians': X.select_dtypes(include=['number']).median().to_dict(),
        # Get mode (most frequent value) for categorical columns
        'modes': X.select_dtypes(include=['object', 'category']).mode().iloc[0].to_dict()
    }
    return stats


def train_model(df, target_column):
  """
  Train a machine learning model on the input DataFrame.
  Args:
    df (pandas.DataFrame): The input DataFrame.
    target_column (str): The name of the target column.
  Returns:
    model (sklearn estimator): The trained machine learning model.
    X (pandas.DataFrame): Features.
    y (pandas.Series): Target variable.
    X_train (pandas.DataFrame): Training features.
    X_test (pandas.DataFrame): Testing features.
    y_train (pandas.Series): Training target variable.
    y_test (pandas.Series): Testing target variable.
    training_stats (dict): A dictionary containing training statistics.
  """

  X = df.drop(columns=[target_column])
  y = df[target_column]

  # Split data into training and testing sets
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

  # Collect training statistics for future use
  train_median = X_train.median()
  train_mode = X_train.mode()
  training_stats = {
      'medians': train_median.to_dict(),
      'modes': train_mode.iloc[0].to_dict()
  }

  model = RandomForestRegressor(
    n_estimators=300,
    min_samples_split=2,
    min_samples_leaf=4,
    max_depth=30,
    bootstrap=True
  )

  model.fit(X_train, y_train)

  return model, X, X_train, X_test, y, y_train, y_test, training_stats

def local_interpretation(model, X_test):
  """
  Perform local interpretation using SHAP values on a trained machine learning model.
  Args:
    model (sklearn estimator): The trained machine learning model.
    X_test (pandas DataFrame): Features used for testing.
  """
  # import matplotlib
  # matplotlib.use('Agg')  # Force non-interactive backend to prevent thread crashes
  # import shap

  random_index = np.random.randint(0, X_test.shape[0])

  #
  instance = X_test.iloc[random_index]

  # Visualize SHAP plot for the selected instance
  print(f"\nGenerating SHAP plot for instance inded: {random_index}")

  # Initialize the explainer
  explainer = shap.TreeExplainer(model)

  shap_values = explainer.shap_values(instance)

  shap.initjs()
  shap.force_plot(
      explainer.expected_value,
      shap_values,
      features=instance,
      matplotlib=True,
      show=False
  )

  # plt.title(f"SHAP Plot for Instance {random_index}", fontsize=16)
  # plt.show()

  explanation = []
  for i, feature in enumerate(X_test.columns.to_list()):
    dollar_impact = shap_values[i] # Fixed: Access shap_values directly
    if abs(dollar_impact) > 1000:
      explanation.append({
          'feature': feature,
          'impact': dollar_impact,
          'contribution': 'increase' if dollar_impact > 0 else 'decrease'
      })

  explanation = sorted(explanation, key=lambda x: abs(x['impact']), reverse=True)

  explanation_df = pd.DataFrame(explanation)
  print(explanation_df.to_string(index=False))


# Pipeline Class
class HousePricePredictor:
  def __init__(self, model, encoders, training_stats, feature_list, ordinal_mapping):
    # Store everything needed for prediction
    self.model = model
    self.encoders = encoders    # saved encoders
    self.training_stats = training_stats    # medians/modes from training
    self.feature_list = feature_list
    self.ordinal_mapping = ordinal_mapping

  def predict(self, data):
    # Add missing features using training_stats
    # aligned_data = self._align_features(data)

    # Fill missing RAW features with training modes/medians
    filled_data = self._fill_missing_raw_features(data)

    # Apply preprocessing using saved encoders
    processed_data = self._apply_encoders(filled_data)

    # Make predictions
    # predictions = self.model.predict(processed_data)

    # Align with model's expected feature list (drop extra cols, add missing OHE cols)
    final_data = self._align_to_model(processed_data)

    return self.model.predict(final_data)

  def _fill_missing_raw_features(self, data):
    df = data.copy()
    # Ensure all raw columns from training_stats exist in the input
    # If missing, fill with the training mode/median
    for col, value in self.training_stats['medians'].items():
      if col not in df.columns:
        df[col] = value
    for col, value in self.training_stats['modes'].items():
      if col not in df.columns:
        df[col] = value
    return df
  
  def _align_features(self, data):
    aligned = pd.DataFrame(columns=self.feature_list)

    for feature in self.feature_list:
      if feature in data.columns:
        aligned[feature] = data[feature]
      else:
        # Use training median/mode
        if feature in self.training_stats['medians']:
          aligned[feature] = self.training_stats['medians'][feature]
        elif feature in self.training_stats['modes']:
          aligned[feature] = self.training_stats['modes'][feature]
        else:
          aligned[feature] = 0

    return aligned

  def _apply_encoders(self, data):
    processed = data.copy()

    # Ordinal mapping
    for col, ratings in self.ordinal_mapping.items():
      if col in processed.columns:
        # Convert to object to handle mixed types, map, then fill NaNs
        processed[col] = processed[col].astype(object).map(ratings).fillna(0)

    # Nominal encoding (OneHot & Target)
    for name, encoder in self.encoders.items():
      if 'onehot' in name:
        col = name.split('_')[1]
        if col in processed.columns:
          # Transform and get new column names
          transformed = encoder.transform(processed[[col]])
          ohe_column_names = encoder.get_feature_names_out([col])

          # Create DataFrame for the new one-hot encoded columns
          if len(ohe_column_names) > 0:
            ohe_df = pd.DataFrame(transformed, index=processed.index, columns=ohe_column_names)

            # Concatenate the new one-hot encoded columns to the original DataFrame
            processed = pd.concat([processed, ohe_df], axis=1)

          # Drop original column after encoding
          processed.drop(columns=[col], inplace=True)
      elif 'target' in name:
        col = name.split('_')[1]
        if col in processed.columns:
          processed[f'{col}_encoded'] = encoder.transform(processed[[col]])
          processed.drop(columns=[col], inplace=True)

    return processed
  
  def _align_to_model(self, data):
    # Create a DataFrame with exactly the columns the model expects
    aligned = pd.DataFrame(index=data.index)
    
    for col in self.feature_list:
      if col in data.columns:
        aligned[col] = data[col]
      else:
        # If an encoded column is missing (e.g. 'Neighborhood_Blueste'), fill with 0
        aligned[col] = 0
        
    # Reorder to match model's expected order
    return aligned[self.feature_list]

  def _explain_prediction(self, data):
    processed = self._apply_encoders(data)
    processed = self._align_to_model(processed)
    explainer = shap.TreeExplainer(self.model)
    shap_values = explainer.shap_values(processed)

    # Handle SHAP output format (sometimes list, sometimes array)
    if isinstance(shap_values, list):
         vals = shap_values[0]
    else:
         vals = shap_values[0] if len(shap_values.shape) > 1 else shap_values

    explanation = []
    for i, feature in enumerate(processed.columns.to_list()):
      dollar_impact = vals[i]
      if abs(dollar_impact) > 1000:
        explanation.append({
            'feature': feature,
            'impact': dollar_impact,
            'contribution': 'increase' if dollar_impact > 0 else 'decrease'
        })

    explanation = sorted(explanation, key=lambda x: abs(x['impact']), reverse=True)

    return explanation

  def _predict_with_explanation(self, data):
    return self.predict(data), self._explain_prediction(data)

# Execution Pipeline
# if __name__ == "__main__":
#   file_path = "train.csv"
  
#   df = load_data(file_path)
  
#   # Target column
#   target_column = 'SalePrice'

#   # Numerical
#   numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns.to_list()
#   numerical_columns.remove(target_column)

#   # Discrete
#   discrete_columns = [
#       'BsmtFullBath', 'BsmtHalfBath',
#       'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr',
#       'TotRmsAbvGrd', 'Fireplaces', 'GarageCars'
#   ]

#   # Temporal
#   temporal_columns = [
#       'YearBuilt', 'YearRemodAdd', 'GarageYrBlt', 'MoSold', 'YrSold'
#   ]

#   # Final Numerical
#   numerical_columns = [col for col in numerical_columns if col not in discrete_columns + temporal_columns]

#   # Ordinal
#   ordinal_columns = [
#       'LotShape', 'Utilities', 'LandSlope', 'OverallQual', 'OverallCond',
#       'ExterQual', 'ExterCond', 'BsmtQual','BsmtCond', 'BsmtExposure', 'BsmtFinType1',
#       'BsmtFinType2', 'HeatingQC', 'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu',
#       'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 'Fence'
#   ]

#   # Nominal
#   nominal_columns = [
#       'MSZoning', 'Street', 'Alley', 'LandContour',
#       'LotConfig', 'Neighborhood', 'Condition1', 'Condition2',
#       'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',
#       'Exterior2nd', 'MasVnrType', 'Foundation', 'Heating', 'CentralAir',
#       'GarageType', 'MiscFeature', 'SaleType', 'SaleCondition'
#   ]

#   # Define Ordinal Mapping
#   ordinal_mapping = {
#       'LotShape': {'IR3': 0, 'IR2': 1, 'IR1': 2, 'Reg': 3},
#       'Utilities': {'None': 0, 'ELO': 1, 'NoSeWa': 2, 'NoSeWr': 3, 'AllPub': 4},
#       'LandSlope': {'None': 0, 'Sev': 1, 'Mod': 2, 'Gtl': 3},
#       'ExterQual': {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
#       'ExterCond': {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
#       'BsmtQual': {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
#       'BsmtCond': {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
#       'BsmtExposure': {'None': 0, 'No': 1, 'Mn': 2, 'Av': 3, 'Gd': 4},
#       'BsmtFinType1': {'None': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6},
#       'BsmtFinType2': {'None': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6},
#       'HeatingQC': {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
#       'Electrical': {'None': 0, 'Mix': 1, 'FuseP': 2, 'FuseF': 3, 'FuseA': 4, 'SBrkr': 5},
#       'KitchenQual': {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
#       'Functional': {'None': 0, 'Sal': 1, 'Sev': 2, 'Maj2': 3, 'Maj1': 4, 'Mod': 5, 'Min2': 6, 'Min1': 7, 'Typ': 8},
#       'FireplaceQu': {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
#       'GarageFinish': {'None': 0, 'Unf': 1, 'RFn': 2, 'Fin': 3},
#       'GarageQual': {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
#       'GarageCond': {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
#       'PavedDrive': {'None': 0, 'P': 1, 'Y': 2},
#       'PoolQC': {'None': 0, 'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4},
#       'Fence': {'None': 0, 'MnWw': 1, 'GdWo': 2, 'MnPrv': 3, 'GdPrv': 4}
#   }

#   missing_summary_df, high_missing_columns = data_exploration(df)

#   columns_to_drop, categorical_imputation, continuous_imputation = imputation_strategy(df, high_missing_columns)

#   df_cleaned, log = clean_data(df, high_missing_columns)

#   df_engineered = engineer_features(df_cleaned)

#   # --- Get stats from the RAW data (before encoding) ---
#   # We use this to fill missing user inputs in the App
#   raw_training_stats = get_raw_training_stats(df_engineered, target_column)

#   # Fix: Pass the actual nominal_columns to analyze_nominal_cardinality
#   cardinality_info, low_cardinality, high_cardinality = analyze_nominal_cardinality(df_engineered, nominal_columns)

#   df_encoded, encoders = create_encoding(df_engineered, ordinal_mapping, low_cardinality, high_cardinality)

#   model, X, X_train, X_test, y, y_train, y_test, training_stats = train_model(df_encoded, target_column)

#   pipeline = HousePricePredictor(
#     model=model,
#     encoders=encoders,
#     training_stats=raw_training_stats,
#     feature_list=X_train.columns.to_list(),
#     ordinal_mapping=ordinal_mapping
#   )

#   save_pipeline_path = "pipeline.pkl"
#   joblib.dump(pipeline, save_pipeline_path)
#   print(f"\nPipeline saved with {len(X_train.columns)} features.")