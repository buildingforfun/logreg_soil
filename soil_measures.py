# All required libraries are imported here for you.
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

'''
1. Read the data into a pandas DataFrame and perform exploratory data analysis
2. Split the data
3. Evaluate feature performance
4. Create the best_predictive_feature variable
'''

# Step 1 - Load + EDA analysis
# Load the dataset
crops = pd.read_csv("soil_measures.csv")
# Check for missing values
print(crops.isna().sum())
# Check how many crops we have, i.e., multi-class target
print(crops.crop.unique())

# Step 2 - Split into feature and target sets
X = crops.drop(columns="crop")
y = crops["crop"]
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Step 3 - Evaluate feature performance
# Create a dictionary to store the model performance for each feature
feature_performance = {}
# Train a logistic regression model for each feature
for feature in ["N", "P", "K", "ph"]:
    log_reg = LogisticRegression(multi_class="multinomial")
    log_reg.fit(X_train[[feature]], y_train)
    y_pred = log_reg.predict(X_test[[feature]])
    
    # Calculate F1 score, the harmonic mean of precision and recall.
    # Precision - how many of the positive predictions were correct.
    # Recall - how many positive samples were correctly identified.
    # Higher the score (0-1) means model achieves high precision and recall.
    # Best used for multi-class classification
    # Could also use balanced_accuracy_score
    f1 = metrics.f1_score(y_test, y_pred, average="weighted")
    
    # Add feature-f1 score pairs to the dictionary
    feature_performance[feature] = f1
print(feature_performance)

# Step 4 - K produced the best F1 score. Store in best_predictive_feature dictionary
best_predictive_feature = {"K": feature_performance["K"]}
