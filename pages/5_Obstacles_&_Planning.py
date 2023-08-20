import streamlit as st
from utils import completion_obstacles_and_planning

def main():
    st.title("Obstacles & Planning")
    st.write("Upload your SMART goal to work with the AI to identify potential obstacles and develop a plan to overcome them.")
    
    st.session_state.llm_output = []
    st.session_state.smart_goals = []
    
    uploaded_file = st.file_uploader("Upload your SMART goal as a .TXT file")
    if uploaded_file:
        smart_goal = uploaded_file.read().decode("utf-8")
        st.session_state.smart_goals.append(smart_goal)
    else:
        st.warning("Please upload your SMART goal to receive recommendations.")
        
    if st.session_state.smart_goals:
        llm_output, cost = completion_obstacles_and_planning(smart_goal=smart_goal)
        
        st.write("Atlas' Response\n\n")
        st.write(llm_output)
        
        st.write("\n\nCost: ", cost)
        
        st.session_state.llm_output.append(llm_output)
        
        st.download_button("Download Atlas' Response", llm_output, 
                           file_name="obstacles_and_plan.txt")
        
if __name__ == "__main__":
    main()