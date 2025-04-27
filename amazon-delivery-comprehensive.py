"""
# Amazon Delivery Time Prediction Model

This notebook builds a model to predict delivery times for Amazon packages and identifies
customer segments based on delivery characteristics. It includes fixes for handling missing values
and comprehensive comments explaining all components and their purpose.
"""

# Import necessary libraries
import pandas as pd  # Data manipulation library
import numpy as np   # Numerical computing library
import seaborn as sns  # Statistical data visualization library based on matplotlib
import matplotlib.pyplot as plt  # Plotting library
from sklearn.model_selection import train_test_split  # For splitting data into train and test sets
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder  # Data preprocessing tools
from sklearn.compose import ColumnTransformer  # For applying different transformers to different columns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor  # Tree-based ensemble models
from sklearn.linear_model import LinearRegression  # Linear regression model
from sklearn.pipeline import Pipeline  # For chaining preprocessing and modeling steps
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # Model evaluation metrics
from sklearn.cluster import KMeans  # K-means clustering algorithm
from sklearn.impute import SimpleImputer  # For handling missing values
import datetime as dt  # Date and time manipulation
import pickle  # For serializing and deserializing Python objects
from math import radians, sin, cos, sqrt, atan2  # For haversine distance calculation

# Set plot style
sns.set(style="whitegrid")  # Sets a white grid background style for all seaborn plots
plt.rcParams['figure.figsize'] = (12, 8)  # Sets default figure size for matplotlib plots
plt.rcParams['font.size'] = 12  # Sets default font size for matplotlib plots

# Load Amazon delivery data
try:
    # Try to load the CSV file
    df = pd.read_csv('amazon_delivery.csv')
    print("Successfully loaded amazon_delivery.csv")
except:
    # Create sample data if file doesn't exist
    print("Could not load amazon_delivery.csv, creating sample data instead...")
    # Sample data with common delivery attributes
    data = {
        'Order_ID': ['ialx566343618', 'akqg208421122', 'njpu434582536', 'rjto796129700'],
        'Agent_Age': [37, 34, 23, 38],
        'Agent_Rating': [4.9, 4.5, 4.4, 4.7],
        'Store_Latitude': [22.74505, 12.91304, 12.91426, 11.00367],
        'Store_Longitude': [75.89247, 77.68324, 77.6784, 76.97649],
        'Drop_Latitude': [22.76505, 13.04304, 12.92426, 11.05367],
        'Drop_Longitude': [75.91247, 77.81324, 77.6884, 77.02649],
        'Order_Date': ['2021-01-01', '2021-01-02', '2021-01-03', '2021-01-04'],
        'Order_Time': ['11:30:00', '19:45:00', '08:30:00', '18:00:00'],
        'Pickup_Time': ['11:45:00', '19:50:00', '08:45:00', '18:10:00'],
        'Weather': ['Sunny', 'Stormy', 'Sandstorms', 'Sunny'],
        'Traffic': ['High', 'Jam', 'Low', 'Medium'],
        'Vehicle': ['motorcycle', 'scooter', 'motorcycle', 'motorcycle'],
        'Area': ['Urban', 'Metropolitian', 'Urban', 'Metropolitian'],
        'Delivery_Time': [120, 165, 130, 105],
        'Category': ['Clothing', 'Electronics', 'Sports', 'Cosmetics']
    }
    df = pd.DataFrame(data)  # Create a pandas DataFrame from the dictionary

# Display basic information about the dataset
print("\nDataset Overview:")
print(f"Number of records: {len(df)}")
print("\nData Types:")
print(df.dtypes)  # Shows the data type of each column
print("\nMissing Values:")
print(df.isnull().sum())  # Counts missing values in each column

# Data preprocessing function
def preprocess_amazon_delivery_data(df):
    """
    Preprocess the Amazon delivery data for analysis and modeling.
    
    This function:
    1. Converts dates and times to appropriate formats
    2. Calculates distance between store and delivery location
    3. Creates time-based features (hour of day, is weekend, etc.)
    4. Handles categorical variables
    5. Creates additional engineered features
    
    Args:
        df: The raw DataFrame containing delivery data
        
    Returns:
        processed_df: The processed DataFrame with additional features
        X: Feature matrix ready for model training
        y: Target variable (Delivery_Time)
        categorical_features: List of categorical feature names
        numeric_features: List of numeric feature names
        label_encoders: Dictionary of label encoders for categorical features
    """
    
    # Create a copy to avoid modifying the original dataframe
    processed_df = df.copy()
    
    # Convert date and time columns to datetime
    # FIX #1: Explicitly set dayfirst=True to fix the date parsing warning
    # pd.to_datetime converts string date representations to datetime objects
    # dayfirst=True tells pandas to interpret dates as day-month-year format
    # errors='coerce' will convert invalid dates to NaT (Not a Time, pandas' version of NaN for dates)
    processed_df['Order_Date'] = pd.to_datetime(processed_df['Order_Date'], dayfirst=True, errors='coerce')
    
    # Convert time strings to minutes since midnight
    # This makes time values numeric and easier to use in machine learning models
    def time_to_minutes(time_str):
        """Convert time string (HH:MM:SS) to minutes since midnight."""
        try:
            hours, minutes, seconds = map(int, time_str.split(':'))
            return hours * 60 + minutes
        except:
            return np.nan  # Return NaN if time format is invalid
    
    processed_df['Order_Minutes'] = processed_df['Order_Time'].apply(time_to_minutes)
    processed_df['Pickup_Minutes'] = processed_df['Pickup_Time'].apply(time_to_minutes)
    
    # Calculate waiting time (time between order and pickup)
    processed_df['Waiting_Time'] = processed_df['Pickup_Minutes'] - processed_df['Order_Minutes']
    # Handle overnight orders (e.g., order at 11:30 PM, pickup at 12:30 AM)
    processed_df.loc[processed_df['Waiting_Time'] < 0, 'Waiting_Time'] = processed_df['Waiting_Time'] + 24*60
    
    # Calculate distance between store and delivery location using haversine formula
    # Haversine formula calculates the great-circle distance between two points on a sphere
    # given their longitudes and latitudes
    def haversine_distance(lat1, lon1, lat2, lon2):
        """
        Calculate the great circle distance between two points 
        on the earth specified in decimal degrees of latitude and longitude.
        """
        R = 6371  # Earth radius in kilometers
        
        # Convert decimal degrees to radians
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        
        distance = R * c  # Distance in kilometers
        return distance
    
    # Apply haversine formula to calculate distance for each delivery
    processed_df['Distance'] = processed_df.apply(
        lambda row: haversine_distance(
            row['Store_Latitude'], row['Store_Longitude'],
            row['Drop_Latitude'], row['Drop_Longitude']
        ),
        axis=1
    )
    
    # Extract day of week and hour of day from order date and time
    # These temporal features can be important for delivery time prediction
    processed_df['Day_of_Week'] = processed_df['Order_Date'].dt.day_name()
    processed_df['Hour_of_Day'] = processed_df['Order_Minutes'] // 60
    
    # Create categorical feature: Is_Rush_Hour
    # Rush hours typically have more traffic and may affect delivery times
    rush_hours_morning = (7 <= processed_df['Hour_of_Day']) & (processed_df['Hour_of_Day'] <= 9)
    rush_hours_evening = (17 <= processed_df['Hour_of_Day']) & (processed_df['Hour_of_Day'] <= 19)
    processed_df['Is_Rush_Hour'] = (rush_hours_morning | rush_hours_evening).astype(int)
    
    # Create binary features: Is_Weekend
    # Weekends may have different traffic patterns than weekdays
    processed_df['Is_Weekend'] = processed_df['Day_of_Week'].isin(['Saturday', 'Sunday']).astype(int)
    
    # Create feature: Delivery_Speed (distance divided by delivery time)
    # This represents how fast the delivery was completed (km/hour)
    processed_df['Delivery_Speed'] = processed_df['Distance'] / (processed_df['Delivery_Time'] / 60)
    
    # Select features for modeling
    features = [
        'Agent_Age', 'Agent_Rating', 'Distance', 'Waiting_Time',
        'Hour_of_Day', 'Is_Rush_Hour', 'Is_Weekend', 'Weather',
        'Traffic', 'Vehicle', 'Area', 'Category'
    ]
    
    # Create feature matrix X and target variable y
    X = processed_df[features].copy()
    y = processed_df['Delivery_Time']  # Target variable: delivery time in minutes
    
    # Identify categorical and numeric features
    categorical_features = ['Weather', 'Traffic', 'Vehicle', 'Area', 'Category']
    numeric_features = [f for f in features if f not in categorical_features]
    
    # Create label encoders for exploration (will use OneHotEncoder in the model pipeline)
    # Label encoding converts categorical text data to numeric values
    # For example: ['Sunny', 'Rainy', 'Sunny'] -> [1, 0, 1]
    label_encoders = {}
    for feature in categorical_features:
        le = LabelEncoder()
        # FIX #2: Handle NaN values in categorical features before encoding
        # LabelEncoder cannot handle NaN values directly, so we replace them with 'Unknown'
        X[feature] = X[feature].fillna('Unknown')
        X[feature] = le.fit_transform(X[feature])
        label_encoders[feature] = le
    
    # Handle hour of day as a cyclical feature using sine and cosine transformation
    # This preserves the cyclic nature of time (e.g., hour 23 is close to hour 0)
    X['Hour_Sin'] = np.sin(2 * np.pi * X['Hour_of_Day'] / 24)
    X['Hour_Cos'] = np.cos(2 * np.pi * X['Hour_of_Day'] / 24)
    
    return processed_df, X, y, categorical_features, numeric_features, label_encoders

# Preprocess the data
processed_df, X, y, categorical_features, numeric_features, label_encoders = preprocess_amazon_delivery_data(df)

# Display the processed dataframe
print("\nProcessed Features:")
print(X.head())  # Show the first few rows of the processed features

# Check for any remaining missing values in X
print("\nMissing Values in Processed Features:")
print(X.isnull().sum())

# FIX #3: If there are still any NaN values in numeric columns, fill them
# This is a safety measure in case any NaNs still exist after preprocessing
for feature in numeric_features:
    if X[feature].isnull().sum() > 0:
        print(f"Filling {X[feature].isnull().sum()} missing values in {feature} with mean")
        # Fill NaN values with the mean of the column
        # Mean imputation is a simple strategy that preserves the average of the feature
        X[feature] = X[feature].fillna(X[feature].mean())

# Split the data into training and testing sets
# test_size=0.2 means 20% of data will be in the test set, 80% in the training set
# random_state ensures reproducibility of the split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# FIX #4: Create preprocessing pipeline with imputation steps
# ColumnTransformer applies different preprocessing to different columns
# Pipeline chains multiple processing steps together
preprocessor = ColumnTransformer(
    transformers=[
        # For numeric features: first impute missing values, then scale
        ('num', Pipeline([
            # SimpleImputer: Fills missing values with a specified strategy
            # strategy='mean': Replace missing values with the mean of the column
            # This ensures no NaN values remain in numeric features
            ('imputer', SimpleImputer(strategy='mean')),
            
            # StandardScaler: Standardizes features by removing the mean and scaling to unit variance
            # This is important for many ML algorithms that assume features are on similar scales
            # Formula: z = (x - mean) / std
            ('scaler', StandardScaler())
        ]), numeric_features),
        
        # For categorical features: first impute missing values, then one-hot encode
        ('cat', Pipeline([
            # SimpleImputer: Fills missing values in categorical features
            # strategy='most_frequent': Replace missing values with the most common value
            # This ensures no NaN values remain in categorical features
            ('imputer', SimpleImputer(strategy='most_frequent')),
            
            # OneHotEncoder: Converts categorical variables into binary vectors
            # For example: ['red', 'green', 'blue'] -> [[1,0,0], [0,1,0], [0,0,1]]
            # handle_unknown='ignore': Ignores unknown categories in new data
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ]), categorical_features)
    ]
)

# Create and train delivery time prediction model
# Pipeline chains preprocessing and model training into a single object
model_pipeline = Pipeline([
    # First step: Apply the preprocessing defined above
    ('preprocessor', preprocessor),
    
    # Second step: Train a RandomForestRegressor model on the preprocessed data
    # RandomForestRegressor: An ensemble learning method that constructs multiple decision trees
    # and outputs the average prediction of individual trees
    # n_estimators=100: Number of trees in the forest
    # random_state=42: Ensures reproducibility
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Fit the model on the training data
# This performs preprocessing and model training in one step
model_pipeline.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model_pipeline.predict(X_test)

# Evaluate the model using multiple metrics
# Mean Absolute Error (MAE): Average absolute difference between predicted and actual values
# Lower is better, unit is the same as the target variable (minutes)
mae = mean_absolute_error(y_test, y_pred)

# Root Mean Squared Error (RMSE): Square root of the average squared differences
# Penalizes larger errors more than MAE, unit is the same as the target variable
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# R² (Coefficient of Determination): Proportion of variance in the dependent variable predictable from the independent variables
# Range: 0 to 1, where 1 is perfect prediction
r2 = r2_score(y_test, y_pred)

print("\nModel Performance:")
print(f"MAE: {mae:.2f} minutes")  # Average error in minutes
print(f"RMSE: {rmse:.2f} minutes")  # Root mean squared error in minutes
print(f"R²: {r2:.2f}")  # R-squared value (proportion of variance explained)

# Feature importance analysis
# Feature importance shows which features have the most impact on predictions
# Get feature names after one-hot encoding
# This code assumes the model has been successfully trained
cat_feature_names = []
ohe = preprocessor.transformers_[1][1].named_steps['encoder']
for i, feature in enumerate(categorical_features):
    if hasattr(ohe, 'categories_'):
        categories = ohe.categories_[i]
        for category in categories:
            cat_feature_names.append(f"{feature}_{category}")
    else:
        # If for some reason categories_ is not available, create a placeholder
        cat_feature_names.append(f"{feature}_unknown")

# Combine all feature names (numeric + transformed hour + categorical)
feature_names = numeric_features + ['Hour_Sin', 'Hour_Cos'] + cat_feature_names

# Extract feature importances (if using a tree-based model)
# RandomForestRegressor calculates feature importances based on how much each feature
# reduces the variance when used in splits across all trees
if hasattr(model_pipeline.named_steps['regressor'], 'feature_importances_'):
    importances = model_pipeline.named_steps['regressor'].feature_importances_
    # Make sure we have the right number of feature names
    if len(importances) == len(feature_names):
        # Sort importances in descending order
        indices = np.argsort(importances)[::-1]
        
        # Plot feature importances
        plt.figure(figsize=(12, 8))
        plt.title('Feature Importance for Delivery Time Prediction')
        plt.bar(range(len(indices)), importances[indices], align='center')
        plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=90)
        plt.tight_layout()
        plt.show()
    else:
        print("Number of feature names doesn't match feature importances. Skipping importance plot.")

# Customer segmentation based on delivery characteristics
# Segmentation helps identify patterns and groups in the data
# Select features for clustering
cluster_features = [
    'Distance', 'Delivery_Time', 'Waiting_Time', 
    'Agent_Rating', 'Delivery_Speed'
]

# FIX #5: Create clustering data with proper handling of missing values
# Create a subset of the processed data with only the clustering features
cluster_data = processed_df[cluster_features].copy()

# Check for missing values in clustering data
print("\nMissing Values in Clustering Data:")
print(cluster_data.isnull().sum())

# Impute missing values in cluster data
# SimpleImputer: Fills missing values with a specified strategy
# strategy='mean': Replace missing values with the mean of the column
# KMeans algorithm cannot handle NaN values, so imputation is necessary
imputer = SimpleImputer(strategy='mean')
cluster_data_imputed = imputer.fit_transform(cluster_data)

# Scale the imputed data
# StandardScaler: Standardizes features by removing the mean and scaling to unit variance
# Scaling is important for KMeans, which uses distance metrics
# Without scaling, features with larger ranges would dominate the clustering
scaler = StandardScaler()
scaled_data = scaler.fit_transform(cluster_data_imputed)

# Find optimal number of clusters using the Elbow method
# The Elbow method plots the sum of squared distances (inertia) for different k values
# The "elbow" in the plot suggests a good number of clusters
inertia = []
k_range = range(1, min(11, len(df)))
for k in k_range:
    # KMeans: Clusters data by trying to separate samples into n groups of equal variance
    # n_clusters=k: Number of clusters to form
    # random_state=42: Ensures reproducibility
    # n_init=10: Number of times the algorithm will run with different centroid seeds
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(scaled_data)
    # inertia_: Sum of squared distances of samples to their closest cluster center
    inertia.append(kmeans.inertia_)

# Plot Elbow method
plt.figure(figsize=(10, 6))
plt.plot(k_range, inertia, 'o-')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia (Within-cluster sum of squares)')
plt.grid(True, alpha=0.3)
plt.show()

# Choose number of clusters (adjust based on the elbow plot)
# We use min(4, len(df)) to handle small datasets
n_clusters = min(4, len(df))
print(f"\nUsing {n_clusters} clusters based on the Elbow method.")

# Perform K-means clustering with the chosen number of clusters
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
# fit_predict: Compute cluster centers and predict cluster index for each sample
clusters = kmeans.fit_predict(scaled_data)

# Add cluster labels to the processed dataframe
processed_df['Cluster'] = clusters

# Analyze clusters by calculating mean values of key metrics for each cluster
cluster_analysis = processed_df.groupby('Cluster').agg({
    'Distance': 'mean',  # Average distance per cluster
    'Delivery_Time': 'mean',  # Average delivery time per cluster
    'Waiting_Time': 'mean',  # Average waiting time per cluster
    'Agent_Rating': 'mean',  # Average agent rating per cluster
    'Delivery_Speed': 'mean',  # Average delivery speed per cluster
    'Order_ID': 'count'  # Number of orders in each cluster
}).rename(columns={'Order_ID': 'count'})

print("\nCluster Analysis:")
print(cluster_analysis)

# Create cluster profiles based on the cluster characteristics
# This gives meaningful names to each cluster based on their statistical properties
cluster_profiles = {}
for cluster in range(n_clusters):
    metrics = cluster_analysis.loc[cluster]
    
    # Interpret clusters based on metrics (delivery time and distance)
    if metrics['Delivery_Time'] < cluster_analysis['Delivery_Time'].mean():
        if metrics['Distance'] < cluster_analysis['Distance'].mean():
            profile = "Quick Local Deliveries"  # Fast deliveries over short distances
        else:
            profile = "Express Long-Distance"  # Fast deliveries over long distances
    else:
        if metrics['Distance'] > cluster_analysis['Distance'].mean():
            profile = "Complex Long-Distance"  # Slow deliveries over long distances
        else:
            profile = "Delayed Local Deliveries"  # Slow deliveries over short distances
    
    cluster_profiles[cluster] = profile

print("\nDelivery Service Profiles:")
for cluster, profile in cluster_profiles.items():
    print(f"Cluster {cluster}: {profile}")

# Define optimization recommendations for each delivery profile
# These recommendations can help improve delivery efficiency for each segment
optimization_recommendations = {}

for cluster, profile in cluster_profiles.items():
    if profile == "Quick Local Deliveries":
        recommendations = [
            "Optimize for batch deliveries in same neighborhood",
            "Use these routes for time-sensitive deliveries",
            "Excellent for promoting 'Delivery in under X minutes' for nearby customers"
        ]
    elif profile == "Express Long-Distance":
        recommendations = [
            "Analyze routes and agent characteristics for successful long-distance deliveries",
            "Prioritize experienced agents with high ratings for these routes",
            "Consider premium pricing for these efficient long-distance deliveries"
        ]
    elif profile == "Complex Long-Distance":
        recommendations = [
            "Identify and address bottlenecks causing delays",
            "Consider alternative routes or transportation methods",
            "Implement better coordination between pickup and delivery stages"
        ]
    elif profile == "Delayed Local Deliveries":
        recommendations = [
            "Investigate local traffic patterns affecting delivery times",
            "Improve dispatch efficiency for shorter distances",
            "Consider micro-fulfillment centers in these areas"
        ]
    else:
        recommendations = [
            "Conduct further analysis to understand this delivery pattern",
            "Compare with historical data to identify optimization opportunities"
        ]
    
    optimization_recommendations[profile] = recommendations

# Function to make predictions for new delivery requests
def predict_delivery_time(new_delivery):
    """
    Predict delivery time for a new delivery request.
    
    This function:
    1. Preprocesses a new delivery request using the same steps as training data
    2. Predicts the delivery time using the trained model
    3. Assigns the delivery to a cluster/profile
    4. Provides optimization recommendations based on the cluster
    
    Args:
        new_delivery: Dictionary with delivery details
        
    Returns:
        predicted_time: Predicted delivery time in minutes
        delivery_profile: Delivery profile/segment
        recommendations: List of optimization recommendations
    """
    # Create a DataFrame for the new delivery
    input_data = pd.DataFrame([new_delivery])
    
    # Preprocess the new data using the same steps as training data
    # Convert time strings to minutes
    def time_to_minutes(time_str):
        hours, minutes, seconds = map(int, time_str.split(':'))
        return hours * 60 + minutes
    
    input_data['Order_Minutes'] = input_data['Order_Time'].apply(time_to_minutes)
    input_data['Pickup_Minutes'] = input_data['Pickup_Time'].apply(time_to_minutes)
    
    # Calculate waiting time
    input_data['Waiting_Time'] = input_data['Pickup_Minutes'] - input_data['Order_Minutes']
    if input_data['Waiting_Time'].iloc[0] < 0:
        input_data['Waiting_Time'] = input_data['Waiting_Time'] + 24*60  # Handle overnight orders
    
    # Calculate distance using haversine formula
    input_data['Distance'] = haversine_distance(
        input_data['Store_Latitude'].iloc[0], input_data['Store_Longitude'].iloc[0],
        input_data['Drop_Latitude'].iloc[0], input_data['Drop_Longitude'].iloc[0]
    )
    
    # Extract hour of day
    input_data['Hour_of_Day'] = input_data['Order_Minutes'] // 60
    
    # Create Is_Rush_Hour feature
    rush_hours_morning = (7 <= input_data['Hour_of_Day']) & (input_data['Hour_of_Day'] <= 9)
    rush_hours_evening = (17 <= input_data['Hour_of_Day']) & (input_data['Hour_of_Day'] <= 19)
    input_data['Is_Rush_Hour'] = (rush_hours_morning | rush_hours_evening).astype(int)
    
    # Create Is_Weekend feature based on the day name
    from datetime import datetime
    order_date = datetime.strptime(input_data['Order_Date'].iloc[0], '%Y-%m-%d')
    day_name = order_date.strftime('%A')
    input_data['Is_Weekend'] = int(day_name in ['Saturday', 'Sunday'])
    
    # Create cyclical hour features
    input_data['Hour_Sin'] = np.sin(2 * np.pi * input_data['Hour_of_Day'] / 24)
    input_data['Hour_Cos'] = np.cos(2 * np.pi * input_data['Hour_of_Day'] / 24)
    
    # Select features for prediction
    pred_features = [
        'Agent_Age', 'Agent_Rating', 'Distance', 'Waiting_Time',
        'Hour_of_Day', 'Is_Rush_Hour', 'Is_Weekend', 'Weather',
        'Traffic', 'Vehicle', 'Area', 'Category'
    ]
    
    # Make sure categorical features use the same encoding
    X_pred = input_data[pred_features].copy()
    
    # Predict delivery time using the trained model pipeline
    # The pipeline automatically handles preprocessing steps
    predicted_time = model_pipeline.predict(X_pred)[0]
    
    # Calculate delivery speed for clustering
    input_data['Delivery_Speed'] = input_data['Distance'] / (predicted_time / 60)  # km/hour
    
    # Create feature vector for clustering
    cluster_features_vector = input_data[cluster_features].values
    
    # Scale the features using the same scaler used for training
    # This ensures consistency with the clustering model
    scaled_vector = scaler.transform(cluster_features_vector)
    
    # Assign to a cluster using the trained KMeans model
    cluster = kmeans.predict(scaled_vector)[0]
    delivery_profile = cluster_profiles[cluster]
    
    # Get recommendations for the assigned cluster/profile
    recommendations = optimization_recommendations[delivery_profile]
    
    return predicted_time, delivery_profile, recommendations

# Example: Make a prediction for a new delivery
new_delivery = {
    'Order_ID': 'new_order_123',
    'Agent_Age': 30,
    'Agent_Rating': 4.8,
    'Store_Latitude': 12.9716,
    'Store_Longitude': 77.5946,
    'Drop_Latitude': 12.9866,
    'Drop_Longitude': 77.6192,
    'Order_Date': '2023-04-15',
    'Order_Time': '12:30:00',
    'Pickup_Time': '12:40:00',
    'Weather': 'Sunny',
    'Traffic': 'Medium',
    'Vehicle': 'motorcycle',
    'Area': 'Urban',
    'Category': 'Electronics'
}

# Make prediction for the new delivery
try:
    predicted_time, delivery_profile, recommendations = predict_delivery_time(new_delivery)
    
    print("\nPrediction for New Delivery:")
    print(f"Predicted Delivery Time: {predicted_time:.1f} minutes")
    print(f"Delivery Profile: {delivery_profile}")
    
    print("\nOptimization Recommendations:")
    for rec in recommendations:
        print(f"- {rec}")
except Exception as e:
    print(f"\nError making prediction: {e}")
    print("This could be due to insufficient training data or mismatched categories.")
    print("In a production environment, we would implement more robust error handling and fallback mechanisms.")

# Save models for future use
# pickle module serializes Python objects to a byte stream
try:
    # Save the model pipeline (includes preprocessing and the regressor)
    with open('amazon_delivery_model.pkl', 'wb') as f:
        pickle.dump(model_pipeline, f)
    
    # Save the clustering model and related components
    with open('amazon_clustering_model.pkl', 'wb') as f:
        pickle.dump((kmeans, scaler, cluster_profiles, optimization_recommendations), f)
    
    print("\nModels saved successfully.")
except Exception as e:
    print(f"\nError saving models: {e}")
