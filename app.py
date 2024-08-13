import validators, streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain.schema import Document
import requests
from bs4 import BeautifulSoup
import json

## Streamlit APP
st.set_page_config(page_title="LangChain: Summarize Text From YT or Website", page_icon="🦜")
st.title("🦜 LangChain: Summarize Text From YT or Website")
st.subheader('Summarize URL')

## Get the Groq API Key and URL (YT or website) to be summarized
st.sidebar.title("Settings")
api_key = st.secrets["GROQ_API_KEY"]
if api_key:
    st.sidebar.success("API Key loaded successfully.")
else:
    st.sidebar.error("API Key not found. Please check your Streamlit secrets")

generic_url = st.text_input("URL", label_visibility="collapsed")

## Gemma Model Using Groq API
llm = ChatGroq(model="Gemma-7b-It", groq_api_key=api_key)

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

def extract_youtube_info(url):
    video_id = url.split("v=")[-1]
    api_url = f"https://www.youtube.com/oembed?url=http://www.youtube.com/watch?v={video_id}&format=json"
    
    try:
        response = requests.get(api_url)
        response.raise_for_status()
        data = response.json()
        
        title = data.get('title', 'Title not found')
        author = data.get('author_name', 'Author not found')
        
        content = f"Title: {title}\nAuthor: {author}\n\nThis is a YouTube video by {author} titled '{title}'."
        return content
    except Exception as e:
        st.error(f"Error extracting YouTube video information: {str(e)}")
        return None

def load_content(url):
    if "youtube.com" in url or "youtu.be" in url:
        content = extract_youtube_info(url)
        if content:
            return [Document(page_content=content)]
        else:
            return None
    else:
        content = scrape_website(url)
        return [Document(page_content=content)]

if st.button("Summarize the Content from YT or Website"):
    if not api_key or not generic_url.strip():
        st.error("Please provide the information to get started")
    elif not validators.url(generic_url):
        st.error("Please enter a valid URL. It can be a YT video URL or website URL")
    else:
        try:
            with st.spinner("Extracting content..."):
                docs = load_content(generic_url)
                
                if not docs or not docs[0].page_content.strip():
                    st.error("Failed to extract content from the provided URL. Please check the URL or try another one.")
                else:
                    st.success("Content extracted successfully!")
                    st.write("Extracted content:")
                    st.text(docs[0].page_content)
                    
                    with st.spinner("Generating summary..."):
                        chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
                        output_summary = chain.run(docs)
                        st.success("Summary generated successfully!")
                        st.write(output_summary)
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
