import validators, streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader
from langchain.schema import Document
import os
import requests
from bs4 import BeautifulSoup

## Streamlit APP
st.set_page_config(page_title="LangChain: Summarize Text From YT or Website", page_icon="🦜")
st.title("🦜 LangChain: Summarize Text From YT or Website")
st.subheader('Summarize URL')

## Get the Groq API Key and URL (YT or website) to be summarized
st.sidebar.title("Settings")
api_key = os.getenv("GROQ_API_KEY")

if api_key:
    st.sidebar.success("API Key loaded from .env file.")
else:
    st.sidebar.error("API Key not found. Please check your .env file.")

generic_url = st.text_input("URL", label_visibility="collapsed")

## Gemma Model Using Groq API
llm = ChatGroq(model="Gemma-7b-It", groq_api_key=api_key)  # '''USE GEMMA FOR BULLETED OUTPUT'''

prompt_template = """
Provide a summary of the following content in 300 words:
Content: {text}
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

def scrape_website(url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"
    }
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Extracting all the text from <p> tags
    paragraphs = soup.find_all('p')
    content = '\n'.join([para.get_text() for para in paragraphs])
    
    return content

if st.button("Summarize the Content from YT or Website"):
    ## Validate all the inputs
    if not api_key.strip() or not generic_url.strip():
        st.error("Please provide the information to get started")
    elif not validators.url(generic_url):
        st.error("Please enter a valid URL. It can be a YT video URL or website URL")
    else:
        try:
            with st.spinner("Waiting..."):
                ## Loading the website or YT video data
                if "youtube.com" in generic_url:
                    loader = YoutubeLoader.from_youtube_url(generic_url, add_video_info=True)
                    docs = loader.load()
                else:
                    content = scrape_website(generic_url)
                    docs = [Document(page_content=content)]

                if not docs or not docs[0].page_content.strip():
                    st.error("Failed to extract content from the provided URL. Please check the URL or try another one.")
                else:
                    ## Chain For Summarization
                    chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
                    output_summary = chain.run(docs)

                    st.success(output_summary)
        except Exception as e:
            st.exception(f"Exception: {e}")
