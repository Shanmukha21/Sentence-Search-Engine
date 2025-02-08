import streamlit as st
import pandas as pd
import fitz  # PyMuPDF for PDFs
import docx
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer


# Function to get the top 5 matching sentences
def get_top_matches(reference_text, input_sentence, top_n=5):
    sentences = reference_text
    all_sentences = sentences + [input_sentence]
    
    vectorizer = TfidfVectorizer().fit_transform(all_sentences)
    cosine_similarities = cosine_similarity(vectorizer[-1], vectorizer[:-1]).flatten()
    
    top_indices = cosine_similarities.argsort()[-top_n:][::-1]
    top_matches = [(sentences[i], cosine_similarities[i]) for i in top_indices]
    return top_matches

def summarize_text(text, sentence_count=5):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LexRankSummarizer()
    summary = summarizer(parser.document, sentence_count)
    return " ".join(str(sentence) for sentence in summary)


# Function to read data from an Excel file
def read_data_from_excel(file):
    df = pd.read_excel(file)
    all_sentences = []
    for column in df.columns:
        all_sentences += df[column].dropna().astype(str).tolist()
    return all_sentences 

# Function to extract text from a PDF file
def read_data_from_pdf(file):
    # doc = fitz.open(file)
    try:
        with fitz.open(stream=file.read(), filetype='pdf') as doc:
            extracted_data = []

            for page in doc:
                text = page.get_text("text")
                paragraphs = re.split(r'\n\s*\n+', text)  # Splitting by blank lines (paragraphs)
                extracted_data.extend([para.strip() for para in paragraphs if para.strip()])  # Remove empty spaces
        return extracted_data 
    
    except Exception as e:
        return f"Error reading PDF: {e}" 

# Function to extract text from a Word file
def read_data_from_word(file):
    doc = docx.Document(file)
    extracted_data = [para.text for para in doc.paragraphs if para.text.strip()]
    return extracted_data  # Return a list instead of DataFrame

# Function to determine file type and extract text accordingly
def extract_text_from_file(file):
    file_type = file.name.split(".")[-1].lower()

    if file_type == "xlsx":
        return read_data_from_excel(file)
    elif file_type == "pdf":
        return read_data_from_pdf(file)
    elif file_type == "docx":
        return read_data_from_word(file)
    else:
        st.error("Unsupported file format! Please upload a PDF, Word, or Excel file.")
        return []

# Streamlit app
st.title('Sentence Similarity Finder')

# Step 1: File Upload
uploaded_file = st.file_uploader("Upload a file (PDF, Word, Excel)", type=["xlsx", "pdf", "docx"])

if uploaded_file is not None:
    # Step 2: Extract text from the uploaded file
    reference_text = extract_text_from_file(uploaded_file)

    if reference_text:  # Now it's always a list, avoiding the DataFrame issue
        st.write("### Extracted Data Preview:")
        preview_df = pd.DataFrame({"Extracted Text": reference_text})  # Convert to DataFrame for preview
        st.write(preview_df.head())  

        # Step 3: Input sentence for similarity search
        input_sentence = st.text_input("Enter a sentence to find similar ones:")

        if input_sentence:
            # Step 4: Get the top 5 matches
            top_matches = get_top_matches(reference_text, input_sentence)

            # Step 5: Display the results
            st.write("### Top 5 Similar Sentences:")
            for i, (sentence, score) in enumerate(top_matches):
                st.write(f"**{i+1}.** {sentence} (Similarity Score: {score:.4f})")

if st.button("Summarize Text"):
    summary = summarize_text(extract_text_from_file)
    st.write("### Summary:")
    st.write(summary)
