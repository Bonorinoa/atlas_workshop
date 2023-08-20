import streamlit as st
import streamlit.components.v1 as components
import streamlit_chat

import logging
import random
import os
import pandas as pd
from typing import Optional, Union, Literal

from utils import completion_smart_goal, chat_smart_goal, memory_to_pandas

# data type for avatar style
AvatarStyle = Literal[ 
    "adventurer", 
    "adventurer-neutral", 
    "avataaars",
    "big-ears",
    "big-ears-neutral",
    "big-smile",
    "bottts", 
    "croodles",
    "croodles-neutral",
    "female",
    "gridy",
    "human",
    "identicon",
    "initials",
    "jdenticon",
    "male",
    "micah",
    "miniavs",
    "pixel-art",
    "pixel-art-neutral",
    "personas",
]

COMPONENT_NAME = "streamlit_chat"

root_dir = os.path.dirname(os.path.abspath(__file__))
build_dir = os.path.join(root_dir, "frontend/build")

_streamlit_chat = components.declare_component(
    COMPONENT_NAME,
    path = build_dir)

# load memory globally
memory_path = "test_long_term_memory.json"
memory_df = memory_to_pandas(memory_path)

smart_gen = memory_df['AI_profiles'][5]

def chat_message_ui(message: str,
                is_user: bool = False, # choose random string from AvatarStyle for default
                seed: Optional[Union[int, str]] = 42,
                key: Optional[str] = None):
    '''
    Streamlit chat frontend style and display
    '''
    avatar_style = "pixel-art-neutral" if is_user else "bottts"

    _streamlit_chat(message=message, seed=seed, 
                    isUser=is_user, avatarStyle=avatar_style, key=key)
    
def init_chat(session_state):
    if not hasattr(session_state, "initialized"):
        session_state.smart_goals = []
        session_state.atlas_output = []
        session_state.user_input = []
        session_state.history_outputs = []
        session_state.history_inputs = []
        session_state.random_id = random.randint(0, 1000)
        session_state.user_report = []
        session_state.generation_cost = []
        session_state.session_data = []
        session_state.first_chat = True
        session_state.initialized = True

def main_chat():
    st.title("Goal Setting")
    st.write("Our AI coach will walk you through the process of translating your goals into actionable steps based on the SMART framework and your perma report. SMART stands for Specific, Measurable, Achievable, Relevant, and Time-bound.\n")

    # Set up the chat interface
    st.subheader("Chat with Atlas")
    
    init_chat(st.session_state)
    
    uploaded_file = st.file_uploader("Upload your survey results as a .TXT file")

    if uploaded_file:
        report = uploaded_file.read().decode("utf-8")
        st.session_state.user_report.append(report)
    else:
        st.warning("Please upload your report to receive recommendations.")
    
    form = st.form(key="user_settings")
    with form:
        if st.session_state.first_chat:
            generate_button = form.form_submit_button("Ask.")
            user_input = st.text_input("Write the goal or activity you would like to work on in a few words (e.g., develop healthier lifestyle habits).", key="chat_input")
            if user_input:
                user_msg = f"Hello, I have the following goal: {user_input}. Can you help me turn this into a SMART goal? Let's work on each component step by step. Please help me address the 'Specific' component first."
                #chat_message_ui(user_msg, is_user=True)
                st.session_state.user_input.append(user_msg)
                st.session_state.history_inputs.append(user_msg)
                
                llm_output, cost = set_smart_goal(smart_gen, report, user_input)
                #chat_message_ui(llm_output, is_user=False)
                st.session_state.atlas_output.append(llm_output)
                st.session_state.history_outputs.append(llm_output)
                st.session_state.generation_cost.append(cost)
                
                st.session_state.first_chat = False
        
        else:
            generate_button = form.form_submit_button("Ask.")
            user_input = st.text_input("Say 'I'm ready' to continue or ask a question about the current component.", 
                                        key="chat_input2")
            
            if generate_button and user_input:
            
                user_msg = user_input
                #chat_message_ui(user_msg, is_user=True)
                st.session_state.user_input.append(user_msg)
                st.session_state.history_inputs.append(user_msg)

                llm_output, cost = chat_smart_goal(smart_gen, report, user_input)
                #chat_message_ui(llm_output, is_user=False)
                st.session_state.atlas_output.append(llm_output)
                st.session_state.history_outputs.append(llm_output)
                st.session_state.generation_cost.append(cost)

                session_data = {
                    "user_input": st.session_state.user_input,
                    "atlas_output": st.session_state.atlas_output,
                    "generation_cost": st.session_state.generation_cost,
                }
                st.session_state.session_data.append(session_data)
            
        for i in range(len(st.session_state.history_outputs)):
            chat_message_ui(st.session_state.history_inputs[i], is_user=True, key=f"{i}_user")
            chat_message_ui(st.session_state.history_outputs[i], is_user=False, key=f"{i}_atlas")

def main_completion():
    st.title("Goal Setting")
    st.write("Our AI coach will walk you through the process of translating your goals into actionable steps based on the SMART framework and your perma report. SMART stands for Specific, Measurable, Achievable, Relevant, and Time-bound.\n")

    st.session_state.user_report = []
    st.session_state.smart_goals = []
    
    uploaded_file = st.file_uploader("Upload your survey results as a .TXT file")

    if uploaded_file:
        report = uploaded_file.read().decode("utf-8")
        st.session_state.user_report.append(report)
    else:
        st.warning("Please upload your report to receive recommendations.")
        
    # text input for defining goal to work on
    user_goal = st.text_input("Write the goal or activity you would like to work on in a few words (e.g., develop healthier lifestyle habits).", key="user_input")
    
    if user_goal and report:
        if st.button("Generate SMART goal"):
            llm_output, cost = completion_smart_goal(smart_gen, 
                                                     report, user_goal)
            st.write("SMART goal: \n\n")
            st.write(llm_output)
            st.write(f"\n\nGeneration cost: {cost}")       
            
            st.session_state.smart_goals.append(llm_output)

if __name__ == "__main__":
    main_completion()
    
    