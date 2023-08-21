from utils import (memory_to_pandas, build_pillar_report, build_pillar_report)
import streamlit as st
import json
import time
import datetime as dt
import pandas as pd
import datetime as dt

# load memory globally
memory_path = "test_long_term_memory.json"
memory_df = memory_to_pandas(memory_path)

# Agents
# 1. Well-being coach
coach = memory_df['AI_profiles'][0]

# 2. Journalist
journalist = memory_df['AI_profiles'][1]

# 3. Recommendationg Engine
recommendator = memory_df["AI_profiles"][2]

# 4. Digital Nudger
nudger = memory_df["AI_profiles"][3]

# 5. Report Generator
report_gen = memory_df["AI_profiles"][4]

# 6. SMART Goal Generator
smart_gen = memory_df["AI_profiles"][5]

# util function to load survey questions given pillar
def load_survey(pillar):
    with open("pillars_q.json") as f:
        survey_questions = json.load(f)
    
    return survey_questions[pillar]


def run_demo():
    st.title("Atlas Intelligence Workshop Demo")
    
    st.write("This is a demo of the Atlas Intelligence Workshop. The workshop is a set of tools that can be used to build AI agents that can help people improve their well-being. The workshop is designed to be used by non-technical users, and is based on the PERMA4 model of well-being. The workshop is currently in beta, and we are looking for feedback on how to improve it. If you have any feedback, please let us know by filling out the feedback form.")

    # user information we want to demo (used only for tuning goal recommendation not report)
    st.sidebar.header("User Profile")
    name = st.sidebar.text_input("Name", "Enter your name")
    age = st.sidebar.text_input("Age", "Enter your age")
    tastes = st.sidebar.text_area("Tastes", "What hobbies or interests do you have?") 
    occupation = st.sidebar.text_input("Occupation", "Enter your occupation")
    location = st.sidebar.text_input("Location", "Where are you based?")
    
    # survey data for selected pillar
    st.sidebar.header("Survey Data")
    pillars = ["Positive Emotions", "Engagement", "Positive Relationships", 
               "Meaning", "Accomplishment", "Physical", "Mindset", "Environment", "Economic"]

    pillar = st.sidebar.selectbox("Select a pillar", pillars)
    
    
    if name and age and tastes and occupation and location and pillar:
        user_data = [name, age, tastes, occupation, location]
        report = ""
        
        # load survey data for selected pillar
        pillar_survey = load_survey(pillar)
        
        instructions = pillar_survey['instructions']
        questions = pillar_survey['questions']
        
        st.subheader(f"Survey questions for {pillar} pillar \n\n")
        
        st.write(instructions)
        
        # Initialize an empty DataFrame to store user responses
        responses_df = pd.DataFrame(columns=["Question", "Response"])
        
        # Use st.form context to handle form submission
        with st.form("survey_form"):
            # Iterate through questions and gather user inputs
            for i, question_obj in enumerate(questions):
                question = question_obj['question']
                question_type = question_obj['type']
                
                st.write(question)  # Display the question
                
                if question_type == 'likert':
                    likert_key = f"likert_{i}"  # Generate a unique key for the slider
                    likert_value = st.slider("Select your answer:", 1, 7, 4, key=likert_key)  # Display likert scale
                    # add response to dataframe
                    responses_df = pd.concat([responses_df, pd.DataFrame.from_records({"Question": question, "Response": likert_value}, index=[0])])
                    
                elif question_type == 'short answer':
                    short_answer_key = f"short_answer_{i}"  # Generate a unique key for the text input
                    short_answer = st.text_area("Your answer:", key=short_answer_key)  # Display text input
                    # add response to dataframe
                    responses_df = pd.concat([responses_df, pd.DataFrame.from_records({"Question": question, "Response": short_answer}, index=[0])])
            

            submit_button = st.form_submit_button("Build Report")  # Submit button
            
            if submit_button:
                st.dataframe(responses_df)
                st.spinner("Building report...")
                
                # Build report for selected pillar
                report, cost = build_pillar_report(report_gen, pillar, 
                                                    responses_df, user_data,
                                                    provider="openai")
                
                st.success("Report built!\n\n")
                #st.write(report)
                st.write(f"{report}")
                #st.write(f"\n\nCost: {cost}")
                st.write(f"Cost: {cost}")
                
        if len(report) > 1:
            st.download_button("Download Report (you will need it to generate activities recommendations)", 
                            data=report, file_name=f"{pillar}_report_{name}_{dt.datetime.today().strftime('%Y-%m-%d')}.txt")   
    else:
        st.warning("Please fill in the user profile information on the left sidebar before generating goals.")
        
    
        
    
if __name__ == "__main__":
    run_demo()