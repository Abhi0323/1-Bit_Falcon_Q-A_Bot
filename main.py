import streamlit as st
import subprocess
import re
import psutil  # To monitor CPU usage
from time import time
from nltk.corpus import stopwords
from PyPDF2 import PdfReader
from docx import Document
import nltk

# Download NLTK stopwords (only once)
nltk.download('stopwords')

# Cache stopwords globally
STOP_WORDS = set(stopwords.words("english"))

# Paths
BITNET_REPO_PATH = "C:\\Users\\AChandragiri\\BitNet"
MODEL_NAME = "models/Falcon3-1B-Instruct-1.58bit/"
INFERENCE_SCRIPT = "run_inference.py"

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# Function to extract text from Word Document
def extract_text_from_docx(docx_file):
    doc = Document(docx_file)
    return "\n".join([p.text for p in doc.paragraphs])

# Function to clean and simplify text
def clean_summary(text):
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation and special characters
    # Use the globally cached STOP_WORDS
    text = " ".join(word for word in text.split() if word.lower() not in STOP_WORDS)  # Remove stopwords
    return text

# Function to extract and clean model response
def clean_model_response(raw_response):
    try:
        match = re.search(r"<\|assistant\|>(.*)", raw_response, re.DOTALL)
        if match:
            response = match.group(1).strip()
            response = response.replace("[end of text]", "").strip()
            return response
        else:
            return "No valid response found."
    except Exception as e:
        return f"Error while processing response: {e}"

# Function to measure CPU usage
def get_cpu_usage():
    return psutil.cpu_percent(interval=0.5)

# Function to run the model and capture stats
def ask_single_question(summary, question):
    try:
        # Construct the structured prompt
        prompt = (
            f"<|system|> You are a helpful assistant. Use the following summary for answering questions:\n"
            f"{summary}\n<|user|> {question}\n<|assistant|>"
        )

        # Construct the command
        command = [
            "python", INFERENCE_SCRIPT,
            "-m", f"{MODEL_NAME}ggml-model-i2_s.gguf",
            "-p", prompt,
            "-n", "128"
        ]

        # Record CPU before running
        initial_cpu = get_cpu_usage()
        start_time = time()

        # Run the subprocess and capture output
        result = subprocess.run(command, capture_output=True, text=True, cwd=BITNET_REPO_PATH)

        # Record time and CPU after running
        end_time = time()
        final_cpu = get_cpu_usage()

        execution_time = round(end_time - start_time, 2)

        # Return response and performance stats
        response = result.stdout.strip() if result.returncode == 0 else f"Error: {result.stderr.strip()}"
        return response, initial_cpu, final_cpu, execution_time
    except Exception as e:
        return f"Error: {e}", 0, 0, 0

# Streamlit App
st.title("1-Bit LLM Implementation & CPU Performance Testing")
st.markdown(
    """
    Experience the power of 1-bit Falcon LLM optimized for low-memory and CPU-efficient inference.
    Upload a document or provide a summary to test question-answering capabilities.
    """
)

# File upload and manual summary input
uploaded_file = st.file_uploader("Upload a PDF or Word document", type=["pdf", "docx"])
manual_summary = st.text_area("Or, enter your text manually", "", height=100)

# Extract text from uploaded file or manual input
document_text = ""
if uploaded_file:
    if uploaded_file.name.endswith(".pdf"):
        document_text = extract_text_from_pdf(uploaded_file)
    elif uploaded_file.name.endswith(".docx"):
        document_text = extract_text_from_docx(uploaded_file)
    else:
        st.error("Unsupported file type.")
elif manual_summary:
    document_text = manual_summary

if document_text:
    # Clean the summary for the initial prompt
    cleaned_summary = clean_summary(document_text)

    # Enter and submit question
    question = st.text_input("Enter your question:")
    if st.button("Submit Question") and question:
        raw_response, initial_cpu, final_cpu, execution_time = ask_single_question(cleaned_summary, question)
        cleaned_response = clean_model_response(raw_response)  # Clean the response

        # Display response
        st.markdown("### Model Response")
        st.text(cleaned_response)

        # Display performance metrics
        st.markdown("### Performance Metrics")
        st.write(f"**CPU Usage Before Execution:** {initial_cpu}%")
        st.write(f"**CPU Usage After Execution:** {final_cpu}%")
        st.write(f"**Time Taken to Generate Response:** {execution_time} seconds")