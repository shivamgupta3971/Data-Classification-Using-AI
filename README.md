# Data Classification Using AI — E-Commerce Order Status

A supervised learning project built as part of the project series. The goal was to build a basic classification model on a small dataset — covering data loading, exploration, train/test splitting, model training, and evaluation, with a bonus comparison across multiple algorithms.

## 📌 Project Goal

Build a basic classification model using a small dataset, demonstrating:
- Data handling
- Supervised learning basics
- Model training and evaluation

## 📂 Dataset

`Dataset_for_Data_Analytics.csv` — a synthetic e-commerce orders dataset with **1,200 rows** and **14 columns**:

| Column | Description |
|---|---|
| OrderID | Unique order identifier |
| Date | Order date |
| CustomerID | Unique customer identifier |
| Product | Product ordered (Monitor, Phone, Tablet, Chair, Laptop, Desk, Printer) |
| Quantity | Units ordered |
| UnitPrice | Price per unit |
| ShippingAddress | Delivery address |
| PaymentMethod | Payment type (Credit Card, Debit Card, Cash, Online, Gift Card) |
| **OrderStatus** | **Target variable** — Cancelled, Returned, Pending, Shipped, Delivered |
| TrackingNumber | Shipment tracking ID |
| ItemsInCart | Number of items in the cart |
| CouponCode | Coupon applied (SAVE10, FREESHIP, WINTER15, or none) |
| ReferralSource | How the customer found the store (Instagram, Email, Google, Facebook, Referral) |
| TotalPrice | Final order total |

The target classes are well balanced (~230–250 rows each), which avoids class-imbalance issues.

## ⚙️ Methodology

1. **Load and understand the dataset** — inspect shape, data types, missing values, class distribution, and summary statistics.
2. **Feature engineering** — parse `Date` into `OrderMonth`/`OrderDayOfWeek`, fill missing `CouponCode` values with `"NoCoupon"`.
3. **Exploratory visualization** — scatter plots (Quantity vs. UnitPrice, ItemsInCart vs. TotalPrice), a boxplot of TotalPrice by OrderStatus, and a class distribution bar chart.
4. **Encoding** — categorical columns (`Product`, `PaymentMethod`, `CouponCode`, `ReferralSource`) and the target (`OrderStatus`) encoded with `LabelEncoder`.
5. **Train/test split** — 80% train / 20% test, stratified to preserve class balance.
6. **Model training** — primary model: **K-Nearest Neighbors** (k=5, Euclidean distance).
7. **Evaluation** — accuracy, precision/recall/F1 per class, confusion matrix.
8. **Prediction on new data** — the trained model predicts the status of three brand-new, hand-crafted orders.
9. **Bonus — algorithm comparison** — KNN is benchmarked against Decision Tree, Naive Bayes, and Support Vector Machine (SVM) on the same train/test split.

## 📊 Results

| Model | Accuracy |
|---|---|
| K-Nearest Neighbors | ~20% |
| Decision Tree | ~22% |
| Naive Bayes | ~19% |
| Support Vector Machine | ~19% |

**Finding:** All four algorithms — despite using very different approaches to classification (distance-based, tree-based, probabilistic, and margin-based) — land close to the ~20% baseline expected from random guessing across 5 classes.

This is a meaningful result, not a failure: when *every* algorithm independently converges on the same low accuracy, it's strong evidence that the input features (product, price, payment method, coupon, referral source, etc.) have little to no real statistical relationship with the final order status in this dataset, rather than any single model being poorly suited to the task. Recognizing and explaining this — instead of chasing a better number — is itself a core supervised learning skill.

## 🖼️ Outputs

- `data_exploration.png` — 4-panel EDA visualization (scatter plots, boxplot, class distribution)
- `confusion_matrix.png` — KNN predictions vs. actual classes
- `algorithm_comparison.png` — accuracy bar chart across all 4 algorithms

## 🛠️ Tech Stack

- Python
- pandas, numpy
- scikit-learn
- matplotlib, seaborn

## ▶️ How to Run

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
python data_classification_ecommerce.py
```

Ensure `Dataset_for_Data_Analytics.csv` is in the same directory as the script.

## 🎓 Skills Demonstrated

- Data handling, cleaning, and feature engineering
- Exploratory data analysis and visualization
- Categorical encoding
- Train/test splitting
- Supervised learning (multi-class classification)
- Model evaluation and cross-algorithm comparison
- Critical interpretation of model results

---
*Project completed as part of the project series.*