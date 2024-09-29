# Streamlit app for heart disease classification 
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
import matplotlib.pyplot as plt

# Function to load dataset from user upload
def load_data(uploaded_file):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        return df
    else:
        st.warning("Please upload a CSV file.")
        return None

# Function to train a Logistic Regression model
def train_model(X_train, y_train):
    model = LogisticRegression(max_iter=500)
    model.fit(X_train, y_train)
    return model

# Streamlit app function
def main():
    st.title("Heart Disease Classification App")
    
    # Upload dataset
    uploaded_file = st.sidebar.file_uploader("Upload your heart disease dataset (CSV format)", type=['csv'])
    
    if uploaded_file:
        # Load data
        data = load_data(uploaded_file)
        
        if data is not None:
            st.write("### Dataset Preview")
            st.write(data.head())
            
            # Select features and target
            st.sidebar.write("### Select Features")
            all_columns = data.columns.tolist()
            features = st.sidebar.multiselect("Select features for the model", all_columns[:-1], default=all_columns[:-1])
            target = st.sidebar.selectbox("Select the target column", all_columns)
            
            if features and target:
                X = data[features]
                y = data[target]
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                # Train model
                model = train_model(X_train, y_train)
                
                # Predict on test data
                y_pred = model.predict(X_test)
                
                # Metrics
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='macro')  # Change average to 'macro', 'micro', or 'weighted'
                recall = recall_score(y_test, y_pred, average='macro')        # Change average to 'macro', 'micro', or 'weighted'
                f1 = f1_score(y_test, y_pred, average='macro')                # Change average to 'macro', 'micro', or 'weighted'
                
                # Display metrics
                st.write("### Model Performance")
                st.write(f"Accuracy: {accuracy:.2f}")
                st.write(f"Precision: {precision:.2f}")
                st.write(f"Recall: {recall:.2f}")
                st.write(f"F1 Score: {f1:.2f}")
                
                # ROC Curve
                st.write("### ROC Curve")
                y_prob = model.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_prob, pos_label=model.classes_[1])
                roc_auc = auc(fpr, tpr)
                
                plt.figure()
                plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
                plt.plot([0, 1], [0, 1], 'k--')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('Receiver Operating Characteristic')
                plt.legend(loc="lower right")
                st.pyplot(plt)
                
                # Input form for user data prediction
                st.write("### Predict on Custom Input")
                user_data = {}
                for feature in features:
                    user_data[feature] = st.number_input(f"Enter {feature}", value=float(X_test[feature].mean()))
                
                # Convert input to dataframe
                user_df = pd.DataFrame([user_data])
                st.write("#### Input Data")
                st.write(user_df)
                
                # Predict on custom input
                if st.button("Predict"):
                    user_prediction = model.predict(user_df)
                    user_prob = model.predict_proba(user_df)[:, 1]
                    st.write(f"Prediction: {'Heart Disease' if user_prediction[0] == 1 else 'No Heart Disease'}")
                    st.write(f"Probability of having heart disease: {user_prob[0]:.2f}")

if __name__ == '__main__':
    main()
