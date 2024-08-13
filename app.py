import validators, streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain.schema import Document
import requests
from bs4 import BeautifulSoup
import json
import re

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

def extract_youtube_info(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        html_content = response.text
        
        # Extract ytInitialPlayerResponse using regex
        match = re.search(r'var ytInitialPlayerResponse = ({.*?});', html_content)
        if not match:
            st.error("Could not find ytInitialPlayerResponse in the page")
            return None
        
        json_str = match.group(1)
        
        # Clean the JSON string
        json_str = re.sub(r',\s*}', '}', json_str)
        json_str = re.sub(r',\s*]', ']', json_str)
        
        # Debug: Print the first 500 characters of the cleaned JSON string
        st.write("Cleaned JSON string (first 500 chars):", json_str[:500])
        
        json_data = json.loads(json_str)
        
        title = json_data['videoDetails']['title']
        description = json_data['videoDetails']['shortDescription']
        
        # Extract transcript if available
        transcript = ""
        if 'playerCaptionsTracklistRenderer' in json_data.get('captions', {}):
            caption_url = json_data['captions']['playerCaptionsTracklistRenderer']['captionTracks'][0]['baseUrl']
            caption_response = requests.get(caption_url)
            caption_soup = BeautifulSoup(caption_response.text, 'html.parser')
            transcript = ' '.join([p.text for p in caption_soup.find_all('p')])
        
        content = f"Title: {title}\n\nDescription: {description}\n\nTranscript: {transcript}"
        return content
    
    except json.JSONDecodeError as json_error:
        st.error(f"JSON Decode Error: {str(json_error)}")
        st.write("Error occurred at position:", json_error.pos)
        st.write("Line and column of error:", json_error.lineno, json_error.colno)
        return None
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
    ## Validate all the inputs
    if not api_key or not generic_url.strip():
        st.error("Please provide the information to get started")
    elif not validators.url(generic_url):
        st.error("Please enter a valid URL. It can be a YT video URL or website URL")
    else:
        try:
            with st.spinner("Waiting..."):
                docs = load_content(generic_url)
                
                if not docs or not docs[0].page_content.strip():
                    st.error("Failed to extract content from the provided URL. Please check the URL or try another one.")
                else:
                    ## Chain For Summarization
                    chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
                    output_summary = chain.run(docs)
                    st.success(output_summary)
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
