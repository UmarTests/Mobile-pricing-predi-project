---
🔹 Author: Mohammad Umar  
🔹 Contact: umar.test.49@gmail.com  
---

### 📌 Section 1: Introduction and Objective  

This project is centered around building a **machine learning classification system** that predicts the **price range category of a mobile phone** based on various technical specifications such as RAM, battery capacity, display resolution, and connectivity features.

📱 **Assumed Client:**  
A mobile phone retailer or e-commerce platform looking to automate pricing tier predictions for new devices listed on their platform.

🧩 **Problem Statement:**  
With a wide variety of smartphones entering the market, it becomes difficult to consistently and objectively assign devices to a price category (e.g., budget, mid-range, premium) based solely on specifications. An automated system helps standardize this process and aids in pricing strategy and inventory decisions.

🎯 **Project Objective:**  
To develop an **end-to-end machine learning model** that classifies mobile phones into **four price ranges** (0 = Low-cost, 1 = Budget, 2 = Mid-range, 3 = Premium) with high accuracy using classification algorithms. The system is deployed via a **Streamlit web app** to allow non-technical users to interact with it easily.

---

### 📊 Section 2: Dataset  

📁 **Dataset Source:**  
Provided as part of an internship task. It resembles datasets commonly found in open repositories such as Kaggle.

🔢 **Dimensions:**  
- **Rows:** 2000  
- **Columns:** 21 (20 input features + 1 target variable)

📌 **Important Input Features:**  
- `battery_power`: Total battery power in mAh  
- `ram`: Random Access Memory in MB  
- `px_height`, `px_width`: Screen resolution (used to compute display quality)  
- `talk_time`: Longest time that a single battery charge will last during a call  
- `sc_h`, `sc_w`: Screen height and width in cm  
- `three_g`, `four_g`, `wifi`, `blue`: Connectivity features  
- `n_cores`: Number of processor cores  
- `m_dep`: Mobile depth (thickness in cm)

🎯 **Target Variable:**  
- `price_range`:  
  - 0 = Low-cost 📉  
  - 1 = Budget 💰  
  - 2 = Mid-range 📱  
  - 3 = Premium 🔥

🧼 **Preprocessing Steps:**  
- Feature scaling using **StandardScaler**  
- New feature engineering (e.g., `screen_to_body`, `display_quality`)  
- Dropped low-impact features such as `fc`, `pc`, `dual_sim` after analysis  

🔍 **Key Observations:**  
- RAM had the highest correlation with price range  
- Battery power and screen resolution were strong indicators  
- Feature `display_quality` was derived from resolution to simplify complexity

---

### ⚙️ Section 3: Design / Workflow  

This project follows a classic supervised classification pipeline:

📦 Data Loading → 🧹 Data Cleaning → 📊 EDA → 🧠 Feature Engineering → 🤖 Model Training → 🔍 Evaluation → 🧪 Tuning → 🌐 Streamlit Deployment

markdown
Copy code

#### ✅ Steps Breakdown:

- **Data Loading & Cleaning**
  - Loaded CSV using pandas  
  - Verified missing values and data types  

- **Exploratory Data Analysis (EDA)**
  - Correlation heatmap for feature relationships  
  - Histogram plots for battery and RAM  
  - Class balance check (evenly distributed across all classes)

- **Feature Engineering**
  - `screen_to_body` = (sc_h * sc_w) / (m_dep + mobile_wt)  
  - `display_quality`: SD, HD, HD+, 4K based on px_height & px_width  
  - Removed redundant or less impactful columns  

- **Model Training & Testing**
  - Tried **Logistic Regression** and **Random Forest Classifier**  
  - Split data: 80% training, 20% test  
  - Applied **StandardScaler** on features  

- **Hyperparameter Tuning**
  - GridSearchCV used to find optimal regularization (C) for Logistic Regression  
  - Used cross-validation (cv=5) to validate model robustness  

- **Model Evaluation**
  - Evaluated using Accuracy, Classification Report, and Confusion Matrix  
  - Compared both models  

- **Deployment**
  - Final model integrated into a **Streamlit UI**  
  - Inputs like RAM, battery, etc. handled via dropdowns and sliders for user ease  
  - Scaler and model are loaded for inference  

---

### 📈 Section 4: Results  

#### 🔍 Final Model Chosen: **Random Forest Classifier**

| Metric        | Logistic Regression | Random Forest |
|---------------|---------------------|---------------|
| Accuracy      | 83.25%              | 80.75%        |
| CV Mean Score | 83.3%               | 80.3%         |
| Model Bias    | High in edge cases  | Better generalization |

🧪 **Test Data Performance (Random Forest):**
- Precision: 92% (Class 0), 78% (Class 1), 65% (Class 2), 88% (Class 3)
- Recall: 92% (Class 0), 80% (Class 1), 74% (Class 2), 76% (Class 3)

📊 **Visuals:**
- Confusion Matrix: Balanced but Class 2 had slight confusion with Class 1/3  
- Feature Importance:  
  - `ram` (56%)  
  - `battery_power`  
  - `screen_to_body`  
  - `talk_time`, `n_cores`, `display_quality`

⚠️ **Misclassifications:**
- Some mid-range phones with higher RAM misclassified as premium  
- Feature scaling and engineered features may need more normalization  

---

### ✅ Section 5: Conclusion  

🎯 **Summary:**  
- Successfully built and deployed a **multi-class classification model**  
- Integrated new features (`screen_to_body`, `display_quality`) to improve predictions  
- Achieved **over 80% accuracy** on test set with Random Forest  

🛠 **Challenges Faced:**
- Logistic Regression showed high train/test accuracy but overfit on price 3  
- Some features had disproportionate influence (like RAM), skewing prediction  
- Streamlit app integration needed careful feature alignment with scaler  

🚀 **Future Improvements:**
- Include categorical features like brand, OS (if available)  
- Ensemble techniques or boosting (XGBoost, LightGBM)  
- Try ordinal encoding for display quality instead of numeric  

📚 **Personal Reflection:**  
This project was an excellent hands-on experience in deploying a full ML pipeline. It taught the importance of **feature impact**, **model interpretability**, and **serving models via intuitive UI** for end users. Even with high accuracy, prediction sanity checks remain crucial to detect edge-case bias.

---

⭐ *Thank you for reading! Feel free to explore the codebase and interact with the Streamlit app.*

