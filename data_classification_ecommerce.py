# """
# Project 2: Data Classification Using AI
# -----------------------------------------
# Goal: Build a basic classification model on a small dataset.

# Dataset : Dataset_for_Data_Analytics_-_Sheet1.csv (e-commerce orders)
# Target  : OrderStatus  (Cancelled / Returned / Pending / Shipped / Delivered)

# Pipeline:
# 1. Load and understand the dataset
# 2. Clean / engineer features
# 3. Split into training and testing sets
# 4. Train a classification algorithm (Decision Tree)
# 5. Evaluate performance (accuracy, confusion matrix, classification report)
# """

# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns

# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder
# from sklearn.tree import DecisionTreeClassifier, plot_tree
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import (
#     accuracy_score,
#     classification_report,
#     confusion_matrix,
# )

# # -------------------------------------------------------------------
# # 1. LOAD AND UNDERSTAND THE DATASET
# # -------------------------------------------------------------------
# DATA_PATH = "Dataset_for_Data_Analytics.csv"
# df = pd.read_csv(DATA_PATH)

# print("=" * 60)
# print("STEP 1: DATASET OVERVIEW")
# print("=" * 60)
# print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
# print("\nColumn types:\n", df.dtypes)
# print("\nMissing values:\n", df.isnull().sum())
# print("\nTarget class balance (OrderStatus):\n", df["OrderStatus"].value_counts())

# # -------------------------------------------------------------------
# # 2. FEATURE ENGINEERING / CLEANING
# # -------------------------------------------------------------------
# # Parse Date into useful numeric parts
# df["Date"] = pd.to_datetime(df["Date"])
# df["OrderMonth"] = df["Date"].dt.month
# df["OrderDayOfWeek"] = df["Date"].dt.dayofweek

# # CouponCode has missing values -> treat missing as "NoCoupon"
# df["CouponCode"] = df["CouponCode"].fillna("NoCoupon")

# # Select features that are meaningful for predicting OrderStatus.
# # We drop identifier columns (OrderID, CustomerID, TrackingNumber,
# # ShippingAddress) since they are unique/near-unique and carry no
# # predictive signal, plus they'd cause leakage/overfitting.
# numeric_features = ["Quantity", "UnitPrice", "ItemsInCart", "TotalPrice",
#             "OrderMonth", "OrderDayOfWeek"]
# categorical_features = ["Product", "PaymentMethod", "CouponCode", "ReferralSource"]

# target_col = "OrderStatus"

# model_df = df[numeric_features + categorical_features + [target_col]].copy()

# # Encode categorical columns
# encoders = {}
# for col in categorical_features:
#     le = LabelEncoder()
#     model_df[col] = le.fit_transform(model_df[col])
#     encoders[col] = le

# # Encode target
# target_encoder = LabelEncoder()
# model_df[target_col] = target_encoder.fit_transform(model_df[target_col])

# X = model_df[numeric_features + categorical_features]
# y = model_df[target_col]

# # -------------------------------------------------------------------
# # 3. TRAIN / TEST SPLIT
# # -------------------------------------------------------------------
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42, stratify=y
# )

# print("\n" + "=" * 60)
# print("STEP 2: TRAIN / TEST SPLIT")
# print("=" * 60)
# print(f"Training samples: {X_train.shape[0]}")
# print(f"Testing samples:  {X_test.shape[0]}")

# # -------------------------------------------------------------------
# # 4. TRAIN A CLASSIFICATION MODEL
# # -------------------------------------------------------------------
# # Primary, simple model: Decision Tree (easy to explain/visualize)
# dt_model = DecisionTreeClassifier(max_depth=6, random_state=42)
# dt_model.fit(X_train, y_train)
# dt_pred = dt_model.predict(X_test)

# # Secondary comparison model: Random Forest (usually stronger baseline)
# rf_model = RandomForestClassifier(n_estimators=200, random_state=42)
# rf_model.fit(X_train, y_train)
# rf_pred = rf_model.predict(X_test)

# # -------------------------------------------------------------------
# # 5. EVALUATE
# # -------------------------------------------------------------------
# print("\n" + "=" * 60)
# print("STEP 3: MODEL EVALUATION")
# print("=" * 60)

# for name, y_pred in [("Decision Tree", dt_pred), ("Random Forest", rf_pred)]:
#     acc = accuracy_score(y_test, y_pred)
#     print(f"\n--- {name} ---")
#     print(f"Accuracy: {acc:.3f}")
#     print(classification_report(
#         y_test, y_pred, target_names=target_encoder.classes_
#     ))

# # Confusion matrix for the Decision Tree (primary model)
# cm = confusion_matrix(y_test, dt_pred)
# plt.figure(figsize=(7, 6))
# sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
#             xticklabels=target_encoder.classes_,
#             yticklabels=target_encoder.classes_)
# plt.title("Confusion Matrix - Decision Tree")
# plt.xlabel("Predicted")
# plt.ylabel("Actual")
# plt.tight_layout()
# plt.savefig("confusion_matrix.png", dpi=150)
# plt.close()

# # Feature importance (Random Forest)
# importances = pd.Series(rf_model.feature_importances_, index=X.columns)
# importances = importances.sort_values(ascending=True)
# plt.figure(figsize=(8, 5))
# importances.plot(kind="barh", color="#4C72B0")
# plt.title("Feature Importance (Random Forest)")
# plt.xlabel("Importance")
# plt.tight_layout()
# plt.savefig("feature_importance.png", dpi=150)
# plt.close()

# # Visualize top of the decision tree
# plt.figure(figsize=(20, 10))
# plot_tree(dt_model, max_depth=3, feature_names=X.columns,
#           class_names=target_encoder.classes_, filled=True, fontsize=8)
# plt.title("Decision Tree (top 3 levels)")
# plt.tight_layout()
# plt.savefig("decision_tree.png", dpi=150)
# plt.close()

# print("\nSaved plots: confusion_matrix.png, feature_importance.png, decision_tree.png")
# print("\nDone.")






# ============================================
# DATA CLASSIFICATION PROJECT
# Using the E-Commerce Orders Dataset
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, 
                             classification_report,
                             confusion_matrix)

# ── Load Dataset ──────────────────────────────
df = pd.read_csv('Dataset_for_Data_Analytics.csv')

# ── Feature Engineering ────────────────────────
# Parse Date into useful numeric parts
df['Date'] = pd.to_datetime(df['Date'])
df['OrderMonth'] = df['Date'].dt.month
df['OrderDayOfWeek'] = df['Date'].dt.dayofweek

# CouponCode has missing values -> treat missing as "NoCoupon"
df['CouponCode'] = df['CouponCode'].fillna('NoCoupon')

print("=" * 50)
print("       E-COMMERCE DATASET - FIRST LOOK")
print("=" * 50)
print(f"\n📊 Shape: {df.shape} (rows, columns)")
print(f"📦 Classes: {df['OrderStatus'].unique()}")
print(f"📝 Columns: {list(df.columns)}")


# ── Basic Statistics ──────────────────────────
print("\n📈 DATASET STATISTICS:")
print("-" * 50)
print(df.describe().round(2))

# ── Check for Missing Values ──────────────────
print("\n🔍 MISSING VALUES:")
print("-" * 50)
print(df.isnull().sum())

# ── Class Distribution ────────────────────────
print("\n📦 CLASS DISTRIBUTION:")
print("-" * 50)
print(df['OrderStatus'].value_counts())

# ── Sample Data ───────────────────────────────
print("\n📋 SAMPLE DATA (First 5 rows):")
print("-" * 50)
print(df.head())


# ── Visualization ─────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('E-Commerce Dataset - Data Exploration', 
             fontsize=16, fontweight='bold')

statuses = df['OrderStatus'].unique()
palette = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFD93D']
colors = dict(zip(statuses, palette))

# Plot 1: Quantity vs UnitPrice
ax1 = axes[0, 0]
for status, color in colors.items():
    mask = df['OrderStatus'] == status
    ax1.scatter(df[mask]['Quantity'],
                df[mask]['UnitPrice'],
                c=color, label=status, alpha=0.6)
ax1.set_xlabel('Quantity')
ax1.set_ylabel('Unit Price')
ax1.set_title('Quantity vs Unit Price')
ax1.legend(fontsize=8)

# Plot 2: ItemsInCart vs TotalPrice
ax2 = axes[0, 1]
for status, color in colors.items():
    mask = df['OrderStatus'] == status
    ax2.scatter(df[mask]['ItemsInCart'],
                df[mask]['TotalPrice'],
                c=color, label=status, alpha=0.6)
ax2.set_xlabel('Items In Cart')
ax2.set_ylabel('Total Price')
ax2.set_title('Items In Cart vs Total Price')
ax2.legend(fontsize=8)

# Plot 3: Feature Distribution
ax3 = axes[1, 0]
df.boxplot(column='TotalPrice', 
           by='OrderStatus', ax=ax3)
ax3.set_title('Total Price by Order Status')
ax3.set_xlabel('Order Status')
plt.sca(ax3)
plt.xticks(rotation=20)

# Plot 4: Class Distribution
ax4 = axes[1, 1]
df['OrderStatus'].value_counts().plot(
    kind='bar', ax=ax4, 
    color=palette
)
ax4.set_title('Class Distribution')
ax4.set_xlabel('Order Status')
ax4.set_ylabel('Count')
plt.sca(ax4)
plt.xticks(rotation=20)

plt.tight_layout()
plt.savefig('data_exploration.png', dpi=150)
plt.show()
print("✅ Visualization saved!")




# ============================================
# ENCODING + SPLITTING THE DATA
# ============================================

# ── Encode categorical features ───────────────
numeric_features = ['Quantity', 'UnitPrice', 'ItemsInCart', 'TotalPrice',
                     'OrderMonth', 'OrderDayOfWeek']
categorical_features = ['Product', 'PaymentMethod', 'CouponCode', 'ReferralSource']

encoders = {}
for col in categorical_features:
    le = LabelEncoder()
    df[col + '_enc'] = le.fit_transform(df[col])
    encoders[col] = le

target_encoder = LabelEncoder()
df['OrderStatus_enc'] = target_encoder.fit_transform(df['OrderStatus'])
class_names = list(target_encoder.classes_)

feature_cols = numeric_features + [c + '_enc' for c in categorical_features]

# ── Prepare Features and Target ───────────────
X = df[feature_cols].values   # Features
y = df['OrderStatus_enc'].values  # Labels (0-4)

print("=" * 50)
print("       DATA SPLITTING")
print("=" * 50)
print(f"\n📦 Total samples: {len(X)}")

# ── Split the Data ────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,      # 20% for testing
    random_state=42,    # For reproducibility
    stratify=y          # Equal class distribution
)

print(f"\n🏋️  Training set:  {len(X_train)} samples (80%)")
print(f"🧪 Testing set:   {len(X_test)} samples  (20%)")

# ── Verify Split Distribution ─────────────────
print("\n📊 CLASS DISTRIBUTION IN SPLITS:")
print("-" * 40)
print(f"{'Class':<15} {'Train':>8} {'Test':>8}")
print("-" * 40)

for i, name in enumerate(class_names):
    train_count = np.sum(y_train == i)
    test_count  = np.sum(y_test == i)
    print(f"{name:<15} {train_count:>8} {test_count:>8}")





# ============================================
# CLASSIFICATION - K-Nearest Neighbors (KNN)
# ============================================

print("=" * 50)
print("    TRAINING THE MODEL (KNN)")
print("=" * 50)

# ── Create the Model ──────────────────────────
"""
KNN Algorithm:
- Looks at K nearest neighbors
- Assigns the majority class
- Simple but effective!
"""
model = KNeighborsClassifier(
    n_neighbors=5,    # Look at 5 nearest points
    metric='euclidean'
)

# ── Train the Model ───────────────────────────
print("\n🏋️  Training model...")
model.fit(X_train, y_train)
print("✅ Model trained successfully!")

# ── Make Predictions ──────────────────────────
print("\n🔮 Making predictions on test set...")
y_pred = model.predict(X_test)
print("✅ Predictions complete!")

# ============================================
# MODEL EVALUATION
# ============================================

print("\n" + "=" * 50)
print("       MODEL EVALUATION RESULTS")
print("=" * 50)

# ── Accuracy Score ────────────────────────────
accuracy = accuracy_score(y_test, y_pred)
print(f"\n🎯 ACCURACY: {accuracy:.2%}")

# ── Detailed Report ───────────────────────────
print("\n📋 CLASSIFICATION REPORT:")
print("-" * 50)
print(classification_report(
    y_test, y_pred,
    target_names=class_names
))

# ── Confusion Matrix ──────────────────────────
print("🔲 CONFUSION MATRIX:")
print("-" * 50)
cm = confusion_matrix(y_test, y_pred)

# Visual Confusion Matrix
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=class_names,
    yticklabels=class_names,
    linewidths=1
)
ax.set_title('Confusion Matrix', 
             fontsize=14, fontweight='bold')
ax.set_ylabel('Actual Label', fontsize=12)
ax.set_xlabel('Predicted Label', fontsize=12)
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150)
plt.show()


# ============================================
# PREDICT NEW DATA
# ============================================

print("\n" + "=" * 50)
print("    PREDICTING NEW ORDER STATUS")
print("=" * 50)

# New order examples: [Quantity, UnitPrice, ItemsInCart, TotalPrice,
#                       OrderMonth, OrderDayOfWeek,
#                       Product_enc, PaymentMethod_enc, CouponCode_enc, ReferralSource_enc]
# Build them using real category values passed through the same encoders
def encode_order(quantity, unit_price, items_in_cart, total_price,
                  month, day_of_week, product, payment_method,
                  coupon_code, referral_source):
    return [
        quantity, unit_price, items_in_cart, total_price, month, day_of_week,
        encoders['Product'].transform([product])[0],
        encoders['PaymentMethod'].transform([payment_method])[0],
        encoders['CouponCode'].transform([coupon_code])[0],
        encoders['ReferralSource'].transform([referral_source])[0],
    ]

new_orders = np.array([
    encode_order(5, 570.62, 7, 2853.10, 1, 2, 'Monitor', 'Debit Card', 'SAVE10', 'Instagram'),
    encode_order(2, 151.35, 3, 302.70, 8, 4, 'Phone', 'Online', 'SAVE10', 'Referral'),
    encode_order(1, 273.19, 5, 273.19, 10, 6, 'Chair', 'Debit Card', 'SAVE10', 'Facebook'),
])

# Predict
predictions = model.predict(new_orders)
probabilities = model.predict_proba(new_orders)

print(f"\n{'Order':<8} {'Prediction':<15} {'Confidence'}")
print("-" * 45)

for i, (pred, prob) in enumerate(
    zip(predictions, probabilities)
):
    confidence = max(prob) * 100
    status = class_names[pred]
    print(f"  #{i+1:<5} {status:<15} {confidence:.1f}%")


# ============================================
# BONUS: COMPARE CLASSIFIERS
# ============================================

from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

classifiers = {
    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
    'Decision Tree':       DecisionTreeClassifier(random_state=42),
    'Naive Bayes':         GaussianNB(),
    'Support Vector':      SVC(random_state=42)
}

print("\n" + "=" * 50)
print("    ALGORITHM COMPARISON")
print("=" * 50)
print(f"\n{'Algorithm':<25} {'Accuracy':>10}")
print("-" * 40)

results = {}
for name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    acc = accuracy_score(y_test, pred)
    results[name] = acc
    print(f"{name:<25} {acc:>9.2%}")

# ── Plot Comparison ───────────────────────────
plt.figure(figsize=(10, 6))
bars = plt.bar(
    results.keys(), 
    results.values(),
    color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'],
    edgecolor='white',
    linewidth=1.5
)

# Add value labels on bars
for bar, val in zip(bars, results.values()):
    plt.text(
        bar.get_x() + bar.get_width()/2,
        bar.get_height() + 0.005,
        f'{val:.1%}',
        ha='center', fontweight='bold'
    )

plt.title('Algorithm Accuracy Comparison', 
          fontsize=14, fontweight='bold')
plt.ylabel('Accuracy')
plt.ylim(0, max(results.values()) + 0.1)
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig('algorithm_comparison.png', dpi=150)
plt.show()