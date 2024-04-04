import streamlit as st
from docx import Document
import pickle
import io
import re
from sklearn.preprocessing import LabelEncoder

# Load the pre-trained model
with open('resume_classification_pipeline1.pkl', 'rb') as file:
    pipeline = pickle.load(file)

# Load the LabelEncoder
with open('label_encoder1.pkl', 'rb') as file:
    label_encoder = pickle.load(file)


# Define preprocessing function
def preprocess_text(text):
    # Lowercasing
    text = text.lower()
    # Remove punctuation and special characters
    text = re.sub(r'[^\w\s]', '', text)
    return text


# Create the Streamlit app
def main():
    st.title("Resume Classifier")

    # Set background color for the whole app
    st.markdown(
        """
        <style>
            body {
                background-color: #f0f2f6;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Upload multiple resumes
    uploaded_files = st.file_uploader("Upload your resumes (DOCX format)", type=["docx"], accept_multiple_files=True)

    if uploaded_files:
        for uploaded_file in uploaded_files:
            # Read the DOCX file
            docx_content = uploaded_file.read()
            doc = Document(io.BytesIO(docx_content))
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"

            # Preprocess the text
            cleaned_text = preprocess_text(text)

            # Classify the resume
            category_encoded = pipeline.predict([cleaned_text])[0]

            # Convert encoded category to original category name
            category = label_encoder.inverse_transform([category_encoded])[0]

            # Display the result for each resume with colored text
            st.subheader(f"Resume Category for {uploaded_file.name}:")
            # Use HTML to style the text with colors
            st.write(f"<span style='color:{'green' if category == 'good' else 'red'}'>{category}</span>",
                     unsafe_allow_html=True)


if __name__ == "__main__":
    main()