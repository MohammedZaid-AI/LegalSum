import os
import streamlit as st
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import HuggingFaceHub
from langchain.chains.summarize import load_summarize_chain
from PyPDF2 import PdfReader
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


load_dotenv()
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")


llm = HuggingFaceHub(
    repo_id="mistralai/Mistral-7B-Instruct-v0.3",
    task="text-generation",
    model_kwargs={
        "max_length": 512,
        "temperature": 0.8
    }
)


tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")


language_codes = {
    "Assamese": "asm_Beng",
    "Awadhi": "awa_Deva",
    "Bengali": "ben_Beng",
    "Bhojpuri": "bho_Deva",
    "Chhattisgarhi": "hne_Deva",
    "Gujarati": "guj_Gujr",
    "Hindi": "hin_Deva",
    "Kannada": "kan_Knda",
    "Magahi": "mag_Deva",
    "Maithili": "mai_Deva",
    "Malayalam": "mal_Mlym",
    "Marathi": "mar_Deva",
    "Manipuri": "mni_Beng",
    "Mizo": "lus_Latn",
    "Tamil": "tam_Taml",
    "Telugu": "tel_Telu",
    "Urdu": "urd_Arab"
}

def extract_text_from_pdfs(files):
    """Extract text from uploaded PDF files."""
    text = ""
    for file in files:
        pdf_reader = PdfReader(file)
        for page in pdf_reader.pages:
            content = page.extract_text()
            if content:
                text += content + "\n\n"
    return text

def split_text(text):
    """Split text into manageable chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )
    return text_splitter.split_text(text)

def indianTrans(text, lang_code):
    """Translate text to the specified Indian language using NLLB."""
    inputs = tokenizer(text, return_tensors="pt")
    translated_tokens = model.generate(**inputs, forced_bos_token_id=tokenizer.convert_tokens_to_ids(lang_code))
    return tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]


summary_prompt = PromptTemplate(
    input_variables=["text"],
    template="""
    You are a helpful assistant that summarizes legal documents. 
    Please provide a clear, concise summary of the following text:
    
    {text}
    
    Summary:
    """
)


summarization_chain = load_summarize_chain(
    llm=llm,
    chain_type="stuff",
    prompt=summary_prompt
)


st.title("Legal Document Summarizer and Translator")
st.write("Upload your legal PDFs to generate a summary in English and a translation in a selected Indian language.")

uploaded_files = st.file_uploader(
    "Upload PDF files",
    type=["pdf"],
    accept_multiple_files=True
)

selected_language = st.selectbox("Select Language for Translation:", list(language_codes.keys()))

if st.button("Summarize and Translate") and uploaded_files:
    with st.spinner("Processing..."):
        try:
            # Extract text from PDFs
            extracted_text = extract_text_from_pdfs(uploaded_files)
            
            if not extracted_text.strip():
                st.warning("No readable text found in the uploaded PDFs.")
            else:
                # Split text into chunks
                chunks = split_text(extracted_text)
                documents = [Document(page_content=chunk) for chunk in chunks]

                # Generate Summary
                response = summarization_chain.run(input_documents=documents)
                summary = response.split("Summary:")[-1].strip() if "Summary:" in response else response.strip()

                # Get the language code
                lang_code = language_codes[selected_language]

                # Translate Summary
                hindi_translation = indianTrans(summary, lang_code)

                # Display Results
                st.subheader("Summary in English:")
                st.write(summary)

                st.subheader(f"Summary in {selected_language}:")
                st.write(hindi_translation)

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
