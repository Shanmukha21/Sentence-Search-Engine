# Sentence-Search-Engine

This project is a **Sentence Search Engine** built using **Streamlit, TF-IDF (Term Frequency-Inverse Document Frequency), and Cosine Similarity**. It allows users to upload text documents (PDF, Word, or Excel), extract textual data, and search for the most similar sentences using NLP techniques.

## Features
- **Supports Multiple File Formats**: PDF, Word (.docx), and Excel (.xlsx).
- **Text Extraction**: Extracts text from uploaded documents.
- **Sentence Similarity Search**: Uses **TF-IDF Vectorization** and **Cosine Similarity** to find the top 5 most similar sentences.
- **Text Summarization**: Implements **LexRank Summarization** to generate concise summaries of extracted text.
- **User-Friendly Interface**: Built with Streamlit for easy interaction.

---

## How It Works
### 1. **File Upload & Text Extraction**
- The user uploads a document.
- Depending on the file type, the program extracts text using:
  - **PyMuPDF (`fitz`)** for PDFs
  - **python-docx (`docx`)** for Word files
  - **pandas (`pd.read_excel`)** for Excel files

### 2. **TF-IDF & Cosine Similarity for Sentence Matching**
Once the text is extracted, the user enters a sentence to search for similar ones. 

#### **TF-IDF (Term Frequency-Inverse Document Frequency)**
TF-IDF is a statistical measure used to evaluate how important a word is in a document relative to a collection of documents (corpus).

The formula consists of two components:
1. **Term Frequency (TF):**
   \[
   TF(t) = \frac{f_t}{N}
   \]
   where:
   - \( f_t \) is the number of times term \( t \) appears in a document
   - \( N \) is the total number of terms in the document

2. **Inverse Document Frequency (IDF):**
   \[
   IDF(t) = \log \left( \frac{N_d}{DF_t} + 1 \right)
   \]
   where:
   - \( N_d \) is the total number of documents
   - \( DF_t \) is the number of documents containing term \( t \)

The final **TF-IDF Score** is computed as:
\[
TF-IDF(t) = TF(t) \times IDF(t)
\]

#### **Cosine Similarity**
Cosine similarity measures the similarity between two text vectors using the cosine of the angle between them:
\[
\text{Cosine Similarity} (A, B) = \frac{A \cdot B}{||A|| \times ||B||}
\]
where:
- \( A \) and \( B \) are TF-IDF vectors of the input sentence and document sentences.
- The result is a score between **0 (least similar)** and **1 (most similar)**.

### 3. **Text Summarization**
- **LexRank Summarization** (graph-based ranking) is used to summarize extracted text.
- It selects the most relevant sentences based on their importance in the document.

---

## Installation & Setup
### **1. Clone the Repository**
```bash
git clone https://github.com/yourusername/sentence-search-engine.git
cd sentence-search-engine
```

### **2. Install Dependencies**
```bash
pip install -r requirements.txt
```

### **3. Run the Application**
```bash
streamlit run app.py
```

---

## Dependencies
- `streamlit`
- `pandas`
- `fitz` (PyMuPDF)
- `python-docx`
- `sklearn`
- `sumy`

---

## Future Improvements
- Add support for **TXT file uploads**
- Improve summarization by integrating **BART or GPT-based models**
- Enhance search efficiency using **FAISS (Facebook AI Similarity Search)**

---

## Author
**[Your Name]**

### **GitHub Repository**
[https://github.com/yourusername/sentence-search-engine](https://github.com/yourusername/sentence-search-engine)
