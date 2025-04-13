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
from sklearn.ensemble import RandomForestClassifier
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
sampled_data = data.sample(n=100, random_state=42)
sampled_data[['battery_power', 'ram']].hist(bins=20, figsize=(10, 5), edgecolor="black")
plt.suptitle("📊 Distribution of Battery Power and RAM", fontsize=16)
plt.show()

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
data['screen_to_body'] = (data['sc_h'] * data['sc_w']) / (data['m_dep'] + data['mobile_wt'])
# Apply log transformation to compress the range
data['screen_to_body'] = np.log1p(data['screen_to_body'])

# Create a new feature: Display Quality (Single Ordinal Column)
def categorize_display(px_height, px_width):
    if px_width >= 3840 and px_height >= 2160:
        return 3  # 4K
    elif px_width >= 2560 and px_height >= 1440:
        return 2  # HD+
    elif px_width >= 1920 and px_height >= 1080:
        return 1  # HD
    else:
        return 0  # SD

data['display_quality'] = data.apply(lambda row: categorize_display(row['px_height'], row['px_width']), axis=1)
data['display_quality'] = data['display_quality'].astype(int)
# Drop px_height and px_width since display_quality replaces them
data.drop(['px_height', 'px_width'], axis=1, inplace=True)

print("\n📌 First 5 rows with new feature 'display_quality' and log-transformed 'screen_to_body':")
print(data[['display_quality', 'screen_to_body']].head())

# Dropping low-impact features (as determined previously)
low_impact_features = ['dual_sim', 'sc_w', 'touch_screen', 'mobile_wt', 'clock_speed', 'int_memory', 'fc', 'pc']
X = data.drop(['price_range'] + low_impact_features, axis=1)
y = data['price_range']

print("\n📌 Class Distribution in Target Variable:")
print(y.value_counts(normalize=True))

# --------------------------------------
# 📌 5. Feature Scaling
# --------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, "scaler.pkl")  # Save scaler for inference

# --------------------------------------
# 📌 6. Split Data into Train & Test Sets
# --------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# --------------------------------------
# 📌 7. Train the Random Forest Classifier
# --------------------------------------
rf_clf = RandomForestClassifier(n_estimators=200, random_state=42)
rf_clf.fit(X_train, y_train)

# Save the trained model
joblib.dump(rf_clf, "random_forest_model.pkl")
print("\n✅ Random Forest Model training completed and saved.")

# --------------------------------------
# 📌 8. Model Evaluation
# --------------------------------------
def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    print(f"\n--- {name} ---")
    print("✅ Accuracy:", accuracy_score(y_test, y_pred))
    print("\n📜 Classification Report:\n", classification_report(y_test, y_pred))
    plt.figure(figsize=(5, 4))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
    plt.title(f"Confusion Matrix for {name}")
    plt.show()

evaluate_model("Random Forest", rf_clf, X_test, y_test)

cv_scores = cross_val_score(rf_clf, X_scaled, y, cv=5)
print("\n📌 Cross-validation scores:", cv_scores)
print("📌 Mean CV score:", np.mean(cv_scores))

# --------------------------------------
# 📌 9. Feature Importance (Random Forest)
# --------------------------------------
importances = rf_clf.feature_importances_
features = X.columns
importance_df = pd.DataFrame({"Feature": features, "Importance": importances})
importance_df = importance_df.sort_values("Importance", ascending=False)
print("\n📌 Feature Importance (Random Forest):")
print(importance_df)
importance_df.plot(kind="barh", x="Feature", y="Importance", figsize=(10, 8))
plt.title("🔍 Random Forest Feature Importances")
plt.xlabel("Importance")
plt.show()
