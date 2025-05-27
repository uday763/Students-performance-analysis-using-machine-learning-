# Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

# Set Streamlit page configuration
st.set_page_config(
    page_title="🎓 Student Performance Predictor",
    layout="wide",
)

# Step 2: Load Dataset
@st.cache_data
def load_data():
    return pd.read_csv(r'C:\Users\Soumyadip Dey\Desktop\Performance\Book1.csv')
data = load_data()

# Step 3: Preprocessing
label_encoders = {}
for column in data.select_dtypes(include='object').columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# Create 'pass' column as target
data['pass'] = data['Final year grade'].apply(lambda x: 1 if x >= 10 else 0)
X = data.drop(['Final year grade', 'pass'], axis=1)
y = data['pass']

# Split and scale data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train models and calculate accuracies
models = {
    'Logistic Regression': LogisticRegression(),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
}
accuracies = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracies[name] = accuracy_score(y_test, y_pred)

# App Header
st.markdown("""
<h1 style='text-align: center; font-size: 65px;'>
🎓 Student Performance Predictor 🎓
</h1>
<br><br>
""", unsafe_allow_html=True)

# Sidebar Inputs
st.sidebar.markdown("""
<h1 style='text-align: left; font-size: 40px;'>
📝 Input Student Attributes :
</h1>
<br><br>
""", unsafe_allow_html=True)

# Input collection with "None"/placeholder default
input_data = {}
input_complete = True

for col in X.columns:
    if col in label_encoders:
        options = ["Select"] + list(label_encoders[col].classes_)
        selected = st.sidebar.selectbox(f"{col}", options, key=col)
        if selected == "Select":
            input_data[col] = None
            input_complete = False
        else:
            input_data[col] = label_encoders[col].transform([selected])[0]
    else:
        val = st.sidebar.slider(f"{col}", int(data[col].min()), int(data[col].max()), int(data[col].mean()), key=col)
        input_data[col] = val

# Model selection and prediction
selected_model = st.sidebar.selectbox("Choose a model", ["Select"] + list(models.keys()))
if selected_model == "Select":
    input_complete = False

predict_button = st.sidebar.button("🔍 Predict Result")

# Main Content
st.subheader("🎯 Prediction Result")

if predict_button:
    if input_complete:
        input_df = pd.DataFrame([input_data])
        input_scaled = scaler.transform(input_df)
        prediction = models[selected_model].predict(input_scaled)
        result = "✅ Pass" if prediction[0] == 1 else "❌ Fail"
        model_accuracy = accuracies[selected_model] * 100
        st.success(result)
        st.info(f"📈 **{selected_model} Accuracy:** {model_accuracy:.2f}%")
    else:
        st.warning("⚠️ Please select all required inputs and choose a model.")
else:
    st.write("Prediction result will appear here after clicking **Predict Result**.")

# Feature Importance
st.subheader("📌 Important Features for Prediction :")
importances = models['Random Forest'].feature_importances_
indices = np.argsort(importances)[::-1]
features = X.columns

fig, ax = plt.subplots(figsize=(8, 4))
sns.barplot(x=importances[indices][:10], y=features[indices][:10], ax=ax)
ax.set_title("Top 10 Feature Importances (Random Forest)", fontsize=13)
ax.set_xlabel("Importance", fontsize=9)
ax.set_ylabel("Features", fontsize=9)
ax.tick_params(axis='both', labelsize=5)
plt.tight_layout()
st.pyplot(fig)

# Pie Chart of Top 5 Features
fig_pie, ax_pie = plt.subplots(figsize=(3, 3))
top_features = features[indices][:5]
top_importances = importances[indices][:5]
ax_pie.pie(top_importances, labels=top_features, autopct='%1.1f%%', startangle=45,
           textprops={'fontsize': 4}, colors=sns.color_palette("Set3", len(top_features)))
ax_pie.axis('equal')
plt.title("Top 5 Feature Importances (Random Forest) - Pie Chart")
st.pyplot(fig_pie)

# 🔽 NEW: Confusion Matrix and ROC Curve
if selected_model != "Select":
    st.subheader("📊 Confusion Matrix and ROC Curve")

    model = models[selected_model]
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    fig_cm, ax_cm = plt.subplots()
    ax_cm.set_title('Confusion Matrix')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Fail', 'Pass'])
    disp.plot(ax=ax_cm)
    st.pyplot(fig_cm)

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    fig_roc, ax_roc = plt.subplots()
    ax_roc.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    ax_roc.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax_roc.set_xlim([0.0, 1.0])
    ax_roc.set_ylim([0.0, 1.05])
    ax_roc.set_xlabel('False Positive Rate')
    ax_roc.set_ylabel('True Positive Rate')
    ax_roc.set_title('Receiver Operating Characteristic (ROC)')
    ax_roc.legend(loc="lower right")
    st.pyplot(fig_roc)
 
 # 🧮 Evaluation Metrics
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    st.subheader("📐 Evaluation Metrics")
    st.markdown(f"- **Precision**: `{precision:.2f}`")
    st.markdown(f"- **Recall**: `{recall:.2f}`")
    st.markdown(f"- **F1-Score**: `{f1:.2f}`")