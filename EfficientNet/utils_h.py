import re
from io import BytesIO
from typing import Any, Dict, List

#import docx2txt
import streamlit as st
#from langchain.chains.qa_with_sources import load_qa_with_sources_chain
#from langchain.docstore.document import Document
#from langchain.llms import OpenAI
#from langchain.text_splitter import RecursiveCharacterTextSplitter
#from langchain.vectorstores import VectorStore
#from langchain.vectorstores.faiss import FAISS
#from openai.error import AuthenticationError
#from pypdf import PdfReader
#
#from knowledge_gpt.embeddings import OpenAIEmbeddings
#from knowledge_gpt.prompts import STUFF_PROMPT


#@st.experimental_memo()
#def parse_docx(file: BytesIO) -> str:
#    text = docx2txt.process(file)
#    # Remove multiple newlines
#    text = re.sub(r"\n\s*\n", "\n\n", text)
#    return text


