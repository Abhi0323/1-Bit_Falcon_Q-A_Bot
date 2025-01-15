# 1-Bit Falcon Q-A Bot

## Overview

The **1-Bit Falcon QA Bot** is a groundbreaking project that showcases the potential of quantizing large language models (LLMs) like Falcon to **1-bit precision**. This optimization significantly reduces memory usage and computational demands, making it ideal for deployment on systems with limited resources. The application tests the performance of the quantized model in **document-based question-answering (QA)** tasks.

The core idea revolves around:

- **Model Quantization**: Converting Falcon LLM to 1-bit precision for efficient inference.
- **Prompt Engineering**: Improving QA accuracy by crafting optimized prompts.
- **Performance Metrics**: Measuring execution time and CPU usage to evaluate resource efficiency.

Built with **Streamlit** for an intuitive user interface, the application uses a custom inference script to interact with the Falcon model and deliver accurate answers based on uploaded document summaries.

---

## Features

- **1-Bit Falcon LLM**: Optimized for low memory and CPU usage without compromising accuracy.
- **Document-Based QA**: Allows users to upload documents (PDFs or Word files) and ask questions based on the content.
- **Prompt Engineering**: Implements techniques to refine and enhance the model's understanding of the question and context.
- **Performance Metrics**: Real-time display of CPU usage and execution time for transparency and analysis.
- **Customizable Deployment**: The app is designed for local usage and can be deployed on platforms like Hugging Face Spaces or GitHub.

---

## Requirements

### Python Libraries

The required libraries are listed in `requirements.txt`. Install them using:

```bash
pip install -r requirements.txt
```

---

## Running the Application Locally

### Step 1: Clone the Repository

```bash
git clone https://github.com/<your-username>/1-Bit-Falcon-QA-Bot.git
cd 1-Bit-Falcon-QA-Bot
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Prepare the Model

Place the 1-bit Falcon model (`ggml-model-i2_s.gguf`) in the `models/` directory. If the directory does not exist, create it:

```bash
mkdir models
mv path_to_model/ggml-model-i2_s.gguf models/
```

### Step 4: Run the Application

```bash
streamlit run app.py
```

---

## Using the Application

1. **Upload Documents**: Use the file uploader to upload PDF or Word documents.
2. **Ask Questions**: Enter your question in the text box.
3. **View Results**: See the model's response and performance metrics (CPU usage and execution time).

---

## Key Highlights

- **Model Quantization**: By converting Falcon to 1-bit precision, the app ensures high efficiency and lower memory consumption.
- **Prompt Engineering**: Improves QA accuracy by crafting structured and context-aware prompts.
- **Performance Metrics**: Transparent display of resource usage (CPU and time).
- **Streamlit Interface**: User-friendly interface for interaction and testing.
- **Extensible**: Open for further optimizations, such as adding support for more models or advanced performance tracking.

---

## Troubleshooting

### Common Issues

- **Model Not Found**: Ensure the model file (`ggml-model-i2_s.gguf`) is correctly placed in the `models/` directory.
- **Dependency Errors**: Verify that all dependencies are installed using the `requirements.txt` file.
- **Streamlit Errors**: Ensure that Streamlit is properly installed and compatible with your Python version.
