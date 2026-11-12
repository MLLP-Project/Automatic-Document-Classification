# Automatic Classification of Document Images

**An AI-powered document image classification system using Microsoft's Document Image Transformer (DiT) to automatically categorize documents based on their visual layout and structure, without relying on OCR.**

## Overview

Organizations process large volumes of documents such as emails, resumes, scientific papers, reports, and advertisements. Manually sorting these documents is time-consuming, inefficient, and prone to human error.

**Document-Classifier** addresses this problem through an end-to-end deep learning pipeline that automatically classifies document images into 10 predefined categories using a fine-tuned Vision Transformer architecture.

Unlike conventional OCR-based approaches, this system focuses on the visual appearance, spatial arrangement, and structural characteristics of documents rather than extracting and interpreting their textual content.

The project uses Microsoft's pretrained **Document Image Transformer (DiT)**, fine-tuned on the Tobacco3482 dataset, and provides a Streamlit-based interface for real-time document classification.

## Key Features

- **OCR-Free Classification:** Classifies documents using visual layout and structural patterns without requiring text extraction.
- **Vision Transformer Architecture:** Utilizes Microsoft's pretrained `microsoft/dit-base` model.
- **10-Class Classification:** Categorizes document images into 10 predefined document classes.
- **Transfer Learning:** Fine-tunes a pretrained DiT model for document-specific classification.
- **Real-Time Inference:** Provides an interactive Streamlit web application for uploading and classifying document images.
- **Confidence Scores:** Displays the predicted document category along with its confidence probability.
- **Local Deployment:** Supports running the classification application locally.

## Dataset

The project uses the **Tobacco3482 dataset**, a document image dataset containing documents belonging to 10 different categories.

- **Dataset:** Tobacco3482
- **Source:** Hugging Face Datasets
- **Dataset identifier:** `maveriq/tobacco3482`
- **Dataset link:** https://huggingface.co/datasets/maveriq/tobacco3482

### Data Preparation

- The dataset is divided into training and testing sets using an 80:20 ratio.
- Stratified splitting is applied to preserve the original class distribution across both sets.
- Images are resized to 224 × 224 pixels.
- Pixel values are normalized using the pretrained model's corresponding normalization parameters.

## Model Architecture

The project uses **Document Image Transformer (DiT)**, specifically Microsoft's `microsoft/dit-base` pretrained model.

DiT is based on the Vision Transformer architecture and processes document images by dividing them into smaller image patches.

Through self-attention mechanisms, the model learns relationships between different regions of a document and captures global visual structures that help distinguish between document categories.

### Model Training

- **Pretrained model:** `microsoft/dit-base`
- **Framework:** PyTorch
- **Model library:** Hugging Face Transformers
- **Training epochs:** 3
- **Training environment:** Google Colab (NVIDIA T4 GPU)
- **Fine-tuning approach:** Transfer Learning
- **Training API:** Hugging Face Trainer API
- **Memory optimization:** Gradient accumulation

The best-performing model checkpoint is saved locally in the `my_document_classifier/` directory.

## Project Architecture

The classification pipeline consists of the following stages:

1. **Image Input:** Accepts a document image as input.
2. **Preprocessing:** Resizes and normalizes the image using Hugging Face AutoImageProcessor.
3. **Patch Embedding:** Divides the image into patches for Vision Transformer processing.
4. **Feature Extraction:** Uses DiT self-attention mechanisms to learn document layout and structural features.
5. **Classification:** A custom classification head predicts one of the 10 document categories.
6. **Output:** Returns the predicted category and its confidence score.

## Performance and Results

The fine-tuned model achieved an overall **94% testing accuracy** across the 10 document categories.

### Observations

- The model demonstrated strong classification performance on structurally distinct documents.
- Emails, advertisements, and news articles achieved particularly high precision and recall, with F1-scores ranging from 0.99 to 1.00.
- The model showed the ability to associate previously unseen, graphics-heavy documents with structurally similar training categories.

### Identified Limitations

The model encounters classification difficulties with dense, text-heavy documents that share similar visual layouts.

The lowest recall scores were observed for:

| Document Category | Recall |
|---|---:|
| Scientific Papers | 0.77 |
| Reports | 0.81 |
| Resumes | 0.83 |

Since the current architecture relies exclusively on visual features and does not incorporate OCR, it may struggle to distinguish documents with similar layouts but different textual content.

## Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| Deep Learning Framework | PyTorch |
| Model Architecture | Vision Transformer (DiT) |
| Pretrained Model | microsoft/dit-base |
| Model Development | Hugging Face Transformers |
| Dataset | Tobacco3482 |
| Data Processing | Hugging Face Datasets |
| Evaluation | Scikit-learn, Evaluate |
| Frontend | Streamlit |
| Development Environment | Google Colab, Jupyter Notebook |
| Deployment | Local Windows Environment |

**Note:** The trained model directory `my_document_classifier/` is intentionally excluded from this repository because it contains the model weights and configuration files.

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/MLLP-Project/Automatic-Document-Classification.git
cd Automatic-Document-Classification
```

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv venv
```

Activate the environment:

**Windows:**
```bash
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install streamlit transformers torch torchvision pillow
```

### 4. Obtain the Trained Model

The trained model weights are not included in this repository.

To obtain the model, either:

- Obtain the trained `my_document_classifier/` directory from the project contributors, or
- Run the training pipeline in `ML_Pipeline.ipynb` to fine-tune the pretrained DiT model and generate the model directory.

Place the resulting `my_document_classifier/` directory in the project root, alongside `app.py`.


### 5. Run the Application

Once the trained model is available locally, execute:

```bash
streamlit run app.py
```

Open the following URL in your browser:

```text
http://localhost:8501
```

Upload a document image through the interface to obtain its predicted category and confidence score.

## Practical Applications

DocuSort AI can be applied in several document-intensive environments:

- **Corporate Mailrooms:** Automatically categorizing incoming correspondence and business documents.
- **Legal Document Management:** Separating reports, correspondence, and other document types for easier archival and retrieval.
- **Historical Archives:** Organizing large collections of digitized legacy documents.
- **Digital Document Management:** Reducing manual effort in document classification and organization.

## Future Scope

Potential improvements and extensions include:

1. **Multimodal Document Understanding:** Integrating OCR with visual features using architectures such as Microsoft's LayoutLM to improve classification of visually similar documents.
2. **Larger Training Dataset:** Expanding the training process using the RVL-CDIP dataset containing approximately 400,000 document images.
3. **Improved Classification:** Addressing confusion between visually similar categories such as scientific papers, reports, and resumes.
4. **Enhanced Deployment:** Extending the application into a more scalable document management and archival solution.

## Contributors

**Anushka Bhagat**
- Developed the end-to-end machine learning pipeline.
- Implemented stratified dataset splitting and fine-tuned the pretrained DiT model.
- Developed and deployed the Streamlit web application for document classification.

**Anushka Banerjee**
- Managed project documentation and GitHub repository organization.
- Conducted quality assurance using real-world document images.
- Designed the project presentation and video demonstration.

**Presentation Video:** https://drive.google.com/file/d/12-fqVQXwFxdyIR70bAb_w2TYr-l0rL8b/view?usp=drive_link

## References

1. [Tobacco3482 Dataset – Hugging Face](https://huggingface.co/datasets/maveriq/tobacco3482)
2. [Document Image Transformer – Microsoft](https://huggingface.co/microsoft/dit-base)
3. [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers/)
4. [Streamlit Documentation](https://docs.streamlit.io/)

---