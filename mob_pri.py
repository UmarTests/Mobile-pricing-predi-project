# --------------------------------------
# 📌 1. Import Required Libraries
# --------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import pickle

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# --------------------------------------
# 📌 2. Load and Explore the Dataset
# --------------------------------------
data = pd.read_csv(r'C:\Users\mohdq\OneDrive\Desktop\internship projects\Mobile pricing.pro\dataset.csv')
print("📌 First 5 rows of the dataset:")
print(data.head())

print("\n📌 Dataset Info:")
print(data.info())

print("\n📌 Summary Statistics:")
print(data.describe())

print("\n📌 Missing Values:")
print(data.isnull().sum())

print("---------------------------------------------------------")

# --------------------------------------
# 📌 3. Exploratory Data Analysis (EDA)
# --------------------------------------
# Sample 100 rows for visualization
sampled_data = data.sample(n=100, random_state=42)

# Histogram for Battery Power and RAM
sampled_data[['battery_power', 'ram']].hist(bins=20, figsize=(10, 5), edgecolor="black")
plt.suptitle("📊 Distribution of Battery Power and RAM", fontsize=16)
plt.show()

# Compute and plot correlation heatmap for important features
selected_features = ['battery_power', 'ram', 'px_height', 'px_width', 'int_memory', 'price_range']
corr_matrix = data[selected_features].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("🔍 Correlation Matrix of Selected Features", fontsize=16)
plt.show()

# --------------------------------------
# 📌 4. Feature Engineering and Preprocessing
# --------------------------------------

# Create a new feature: Screen to Body Ratio
# (Original formula: (sc_h * sc_w) / (m_dep + mobile_wt))
data['screen_to_body'] = (data['sc_h'] * data['sc_w']) / (data['m_dep'] + data['mobile_wt'])

# Apply a log1p transformation to screen_to_body to compress its scale
# (np.log1p(x) computes log(1+x))
data['screen_to_body'] = np.log1p(data['screen_to_body'])

# Creating a new feature: Display Quality (Single Ordinal Column)
def categorize_display(px_height, px_width):
    if px_width >= 3840 and px_height >= 2160:
        return 3  # 4K
    elif px_width >= 2560 and px_height >= 1440:
        return 2  # HD+
    elif px_width >= 1920 and px_height >= 1080:
        return 1  # HD
    else:
        return 0  # SD

# Apply function to create the display_quality feature
data['display_quality'] = data.apply(lambda row: categorize_display(row['px_height'], row['px_width']), axis=1)
data['display_quality'] = data['display_quality'].astype(int)

# Drop px_height and px_width since display_quality replaces them
data.drop(['px_height', 'px_width'], axis=1, inplace=True)

print("\n📌 First 5 rows with new feature 'display_quality' and log-transformed 'screen_to_body':")
print(data[['display_quality', 'screen_to_body']].head())

# Dropping low-impact features based on coefficient analysis
low_impact_features = ['dual_sim', 'sc_w', 'touch_screen', 'mobile_wt', 'clock_speed', 'int_memory', 'fc', 'pc']
X = data.drop(['price_range'] + low_impact_features, axis=1)  # Features
y = data['price_range']  # Target variable



# Ensure class balance in the dataset
print("\n📌 Class Distribution in Target Variable:")
print(y.value_counts(normalize=True))  # Shows percentage of each class

# Feature Scaling (IMPORTANT: Save scaler for inference)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save the scaler to use in Streamlit app
joblib.dump(scaler, "scaler.pkl")

# --------------------------------------
# 📌 5. Split Data into Train & Test Sets
# --------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# --------------------------------------
# 📌 6. Train the Logistic Regression Model
# --------------------------------------
from sklearn.linear_model import LogisticRegression

# Apply stronger regularization
log_reg = LogisticRegression(multi_class="multinomial", solver="lbfgs", C=100,max_iter=500)
log_reg.fit(X_train, y_train)


# Save the trained model
joblib.dump(log_reg, "logistic_regression_model.pkl")
print("\n✅ Model training completed and saved.")

# --------------------------------------
# 📌 7. Model Evaluation
# --------------------------------------
def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    print(f"\n--- {name} ---")
    print("✅ Accuracy:", accuracy_score(y_test, y_pred))
    print("\n📜 Classification Report:\n", classification_report(y_test, y_pred))
    
    # Confusion Matrix
    plt.figure(figsize=(5, 4))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
    plt.title(f"Confusion Matrix for {name}")
    plt.show()

# Evaluate Logistic Regression
evaluate_model("Logistic Regression", log_reg, X_test, y_test)

# --------------------------------------
# 📌 8. Cross-Validation & Hyperparameter Tuning
# --------------------------------------
cv_scores = cross_val_score(log_reg, X_scaled, y, cv=5)
print("\n📌 Cross-validation scores:", cv_scores)
print("📌 Mean CV score:", np.mean(cv_scores))

# Hyperparameter tuning
param_grid = {'C': [0.1, 1, 10, 100]}
grid_search = GridSearchCV(LogisticRegression(max_iter=5000, solver='saga', random_state=42), param_grid, cv=5)
grid_search.fit(X_train, y_train)
print("\n✅ Best Hyperparameters:", grid_search.best_params_)
print("✅ Best CV Score:", grid_search.best_score_)

from sklearn.metrics import classification_report

y_pred_train = log_reg.predict(X_train)
y_pred_test = log_reg.predict(X_test)

print("\n📌 Classification Report (Train Data):")
print(classification_report(y_train, y_pred_train))

print("\n📌 Classification Report (Test Data):")
print(classification_report(y_test, y_pred_test))


# --------------------------------------
# 📌 9. Feature Importance (Coefficient Analysis)
# --------------------------------------
# Create DataFrame with coefficients
coef_df = pd.DataFrame(
    log_reg.coef_.T,
    index=X.columns,
    columns=[f"Coefficient_Class_{i}" for i in range(log_reg.coef_.shape[0])]
)

print("\n📌 Logistic Regression Coefficients:")
print(coef_df)

# Visualize feature importance
coef_df["Coefficient_Class_0"].sort_values(ascending=True).plot(kind="barh", figsize=(10, 8))
plt.title("🔍 Feature Coefficients for Class 0")
plt.xlabel("Coefficient Value")
plt.show()

# --------------------------------------
# 📌 10. Save the Model for Deployment
# --------------------------------------
with open("logistic_model.pkl", "wb") as file:
    pickle.dump(log_reg, file)

print("\n✅ Logistic Regression Model Saved Successfully.")

import joblib

# Save the trained StandardScaler
joblib.dump(scaler, "scaler.pkl")
print("✅ Scaler saved as scaler.pkl")


import pandas as pd

# Create a DataFrame with coefficients for all classes
coef_df = pd.DataFrame(
    log_reg.coef_.T,  # Transpose to match features
    index=X.columns,
    columns=[f"Class_{i}" for i in range(log_reg.coef_.shape[0])]
)

# Print coefficients
print("\n📌 Feature Importance (Logistic Regression Coefficients):")
print(coef_df)

# (Optional) Plot feature importance for one class (e.g., Class_3)
coef_df["Class_3"].sort_values(ascending=True).plot(kind="barh", figsize=(10, 8))
plt.title("Feature Coefficients for Class 3 (High-end Phones)")
plt.xlabel("Coefficient Value")
plt.show()
