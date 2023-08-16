import streamlit as st
from utils import generate_goals, suggest_activities, memory_to_pandas
import time
import datetime as dt
from io import StringIO

# load memory globally
memory_path = "test_long_term_memory.json"
memory_df = memory_to_pandas(memory_path)

# Well-being coach LLM profile
coach = memory_df['AI_profiles'][0]

# Recommendation Engine LLM profile
recommendator = memory_df["AI_profiles"][2]

def main():
    st.title("Activity Recommendation Engine")
    st.markdown("Please upload your survey results as a .TXT file to receive your personalized activity recommendations. \nThen, copy the text of one of the recommendations and paste it into the Goal Setting tool to develop a SMART goal for such activtiy.")
    
    # file upload
    uploaded_file = st.file_uploader("Upload your survey results as a .TXT file")
    
    if uploaded_file:
        # parse text file
        stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
        report = stringio.read()
        
        # build goal generator
        goals, g_cost = generate_goals(recommendator, report, 
                                       provider='openai')
        st.spinner("Generating goals...")
        st.write(f"Goals: \n\n {goals}\n\n")
        st.write(f"Cost: {g_cost} \n\n")
        
        # recommend goals to user
        activities, a_cost = suggest_activities(coach, report, goals, 
                                                provider='openai')
        st.spinner("Generating recommendations...")
        
        # display recommendations
        st.write("Recommendations\n\n")
        st.write(f"Goals: \n\n {activities}\n\n")
        
        st.write(f"\n\nCost: {a_cost}")
        
    else:
        st.warning("Please upload your report to receive recommendations.")
        
if __name__ == "__main__":
    main()