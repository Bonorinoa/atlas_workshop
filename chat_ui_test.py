import os
from transformers import pipeline

from langchain.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.tools import wolfram_alpha
from langchain.agents import load_tools, initialize_agent

import streamlit as st

os.environ["WOLFRAM_ALPHA_APPID"] = "42V6VG-3TEWJ62W3Y"
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"


st.set_page_config(page_title="🤗💬 SmartBot Test")


with st.sidebar:
    st.title('🤗💬 SmartBot')
    st.subheader('Powered by 🤗 Language Models')
            
# Store LLM generated responses
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

# Display or clear chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]
st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

# Function for generating LLM response
def generate_response(prompt_input):

    # instantiate llm
    # description and hot to use T5 -> https://huggingface.co/google/flan-t5-base
    model = "google/flan-t5-base"
    llm = HuggingFaceHub(repo_id=model,
                     model_kwargs={"temperature": 0.9,
                                   "max_length": 100})

    for dict_message in st.session_state.messages:
        string_dialogue = "You are a helpful assistant."
        if dict_message["role"] == "user":
            string_dialogue += "User: " + dict_message["content"] + "\n\n"
        else:
            string_dialogue += "Assistant: " + dict_message["content"] + "\n\n"

    prompt = f"{string_dialogue} {prompt_input} Assistant: "
    return llm(prompt)

# User-provided prompt
if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = generate_response(prompt) 
            st.write(response) 
    message = {"role": "assistant", "content": response}
    st.session_state.messages.append(message)
