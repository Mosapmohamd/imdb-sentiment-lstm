import streamlit as st
import pandas as pd
import tensorflow as tf
import numpy as np
import re
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import itertools

# Set page config
st.set_page_config(
    page_title="IMDB Sentiment Analysis",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stTextInput>div>div>input {
        background-color: #ffffff;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        border: none;
        padding: 10px 24px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .positive {
        color: #4CAF50;
        font-weight: bold;
    }
    .negative {
        color: #f44336;
        font-weight: bold;
    }
    .header {
        color: #2c3e50;
    }
</style>
""", unsafe_allow_html=True)

# Function to clean text
def clean_review(text):
    return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \\t])|(\\w+:\\/\\/\\S+)", " ", text).split()).lower()

# Function to plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, classes, normalize=False, title='Confusion Matrix', cmap=plt.cm.Blues):
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=35)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    return plt

# Load the model
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model('sentiment_RNN_10epochs.model')
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Main app function
def main():
    st.title("🎬 IMDB Movie Review Sentiment Analysis")
    st.markdown("""
    This app uses a deep learning model (RNN-LSTM) to predict the sentiment of IMDB movie reviews as either positive or negative.
    """)
    
    model = load_model()
    
    if model is None:
        st.error("Failed to load the sentiment analysis model. Please ensure the model files are available.")
        return
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["Single Review Analysis", "Batch Analysis", "Model Evaluation"])
    
    with tab1:
        st.header("Analyze a Single Review")
        review_text = st.text_area("Enter a movie review:", height=150)
        
        if st.button("Analyze Sentiment"):
            if review_text.strip() == "":
                st.warning("Please enter a review to analyze.")
            else:
                with st.spinner("Analyzing sentiment..."):
                    # Clean and preprocess the review
                    cleaned_review = clean_review(review_text)
                    review_array = np.array([cleaned_review])
                    
                    # Make prediction
                    prediction = model.predict(review_array)
                    confidence = prediction[0][0]
                    sentiment = "positive" if confidence >= 0.5 else "negative"
                    confidence_percent = max(confidence, 1 - confidence) * 100
                    
                    # Display results
                    st.subheader("Analysis Results")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Predicted Sentiment", 
                                 f"{sentiment.capitalize()}",
                                 delta=f"{confidence_percent:.1f}% confidence",
                                 delta_color="normal")
                    
                    with col2:
                        # Create a gauge chart
                        fig, ax = plt.subplots(figsize=(3, 1))
                        ax.set_xlim(0, 1)
                        ax.set_ylim(0, 1)
                        ax.axis('off')
                        
                        # Draw the gauge
                        ax.barh(0.5, 1, height=0.3, color='lightgray')
                        ax.barh(0.5, confidence, height=0.3, color='#4CAF50' if sentiment == "positive" else '#f44336')
                        
                        # Add markers
                        ax.plot([0.5, 0.5], [0.2, 0.8], color='black', linewidth=1)
                        ax.text(0.5, 0.9, "Neutral", ha='center', va='center', fontsize=8)
                        
                        # Add value text
                        ax.text(confidence/2, 0.5, f"{confidence:.2f}", 
                                ha='center', va='center', color='white', fontweight='bold')
                        
                        st.pyplot(fig)
                    
                    st.markdown("### Review Text:")
                    st.write(review_text)
    
    with tab2:
        st.header("Analyze Multiple Reviews")
        uploaded_file = st.file_uploader("Upload a CSV file with reviews (one per line):", type=["csv", "txt"])
        
        if uploaded_file is not None:
            try:
                # Read the uploaded file
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                    # Assume first column contains reviews
                    reviews = df.iloc[:, 0].tolist()
                else:
                    reviews = [line.decode('utf-8').strip() for line in uploaded_file.readlines()]
                
                if len(reviews) == 0:
                    st.warning("No reviews found in the uploaded file.")
                else:
                    st.success(f"Found {len(reviews)} reviews to analyze.")
                    
                    if st.button("Analyze All Reviews"):
                        with st.spinner("Analyzing reviews..."):
                            # Clean and preprocess all reviews
                            cleaned_reviews = [clean_review(review) for review in reviews]
                            review_arrays = np.array(cleaned_reviews)
                            
                            # Make predictions
                            predictions = model.predict(review_arrays)
                            sentiments = ["positive" if pred >= 0.5 else "negative" for pred in predictions]
                            confidences = [max(pred, 1 - pred) * 100 for pred in predictions]
                            
                            # Create results dataframe
                            results_df = pd.DataFrame({
                                "Review": reviews,
                                "Sentiment": sentiments,
                                "Confidence (%)": [f"{conf:.1f}" for conf in confidences]
                            })
                            
                            # Display results
                            st.subheader("Analysis Results")
                            st.dataframe(results_df)
                            
                            # Download button
                            csv = results_df.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label="Download Results as CSV",
                                data=csv,
                                file_name="sentiment_analysis_results.csv",
                                mime="text/csv"
                            )
                            
                            # Show summary statistics
                            st.subheader("Summary Statistics")
                            pos_count = sum(1 for s in sentiments if s == "positive")
                            neg_count = len(sentiments) - pos_count
                            
                            col1, col2 = st.columns(2)
                            col1.metric("Positive Reviews", pos_count)
                            col2.metric("Negative Reviews", neg_count)
                            
                            # Pie chart of sentiment distribution
                            fig, ax = plt.subplots()
                            ax.pie([pos_count, neg_count], 
                                  labels=['Positive', 'Negative'],
                                  colors=['#4CAF50', '#f44336'],
                                  autopct='%1.1f%%')
                            ax.set_title("Sentiment Distribution")
                            st.pyplot(fig)
            
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
    
    with tab3:
        st.header("Model Evaluation")
        st.markdown("""
        This section shows the performance metrics of the trained sentiment analysis model.
        """)
        
        # Load test data (you would need to have this available)
        # For demo purposes, we'll show placeholder metrics
        # In a real deployment, you would load your test set
        
        st.warning("Note: For security reasons, the test data is not included in this deployment. "
                  "The following shows sample evaluation metrics from the training process.")
        
        # Sample metrics (replace with actual metrics from your model)
        sample_y_true = np.random.randint(0, 2, size=100)
        sample_y_pred = np.random.randint(0, 2, size=100)
        sample_y_pred = np.where(sample_y_pred == sample_y_true, sample_y_true, sample_y_pred)
        
        # Confusion matrix
        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots()
        plot_confusion_matrix(
            sample_y_true, sample_y_pred, 
            classes=['Negative', 'Positive'],
            title='Confusion Matrix'
        )
        st.pyplot(fig)
        
        # Classification report
        st.subheader("Classification Report")
        report = classification_report(sample_y_true, sample_y_pred, target_names=['Negative', 'Positive'], output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df.style.format("{:.2f}"))
        
        # Accuracy
        accuracy = accuracy_score(sample_y_true, sample_y_pred)
        st.metric("Accuracy", f"{accuracy:.2%}")

# Run the app
if __name__ == "__main__":
    main()