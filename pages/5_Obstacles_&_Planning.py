import streamlit as st
from utils import completion_obstacles_and_planning, send_email

def main():
    st.title("Obstacles & Planning")
    st.write("Upload your SMART goal to work with the AI to identify potential obstacles and develop a plan to overcome them.")
    
    st.session_state.llm_output = []
    st.session_state.goals = []
    st.session_state.smart_goals = []
    
    uploaded_file = st.file_uploader("Upload your SMART goal as a .TXT file")
    if uploaded_file:
        goal = uploaded_file.name
        st.session_state.goals.append(goal)
        
        smart_goal = uploaded_file.read().decode("utf-8")
        st.session_state.smart_goals.append(smart_goal)
                
    else:
        st.warning("Please upload your SMART goal to receive recommendations.")
        
        
    if st.session_state.smart_goals and goal:
        llm_output, cost = completion_obstacles_and_planning(goal=goal, 
                                                             smart_goal=smart_goal)
        
        st.write("Atlas' Response\n\n")
        st.write(llm_output)
        
        st.write("\n\nCost: ", cost)
        
        st.session_state.llm_output.append(llm_output)
        
        with st.form("Discussion"):
            st.write("What are the main internal and external obstacles that could prevent you from reaching your goal? What are the potential solutions to those obstacles? Write your main obstacles and how you plan to overcome/prevent them.")
            user_answer = st.text_area("Main obstacles and solutions", key="user_input2")
            submit_button = st.form_submit_button(label="Submit")
            if submit_button:
                send_email(user_name="Atlas", user_email="atlas.intelligence21@gmail.com",
                           feedback=user_answer)
                st.success("Your response has been submitted.")
        
        st.download_button("Download Atlas' Response", llm_output + "\n\n--\n\n" + user_answer, 
                           file_name="obstacles_and_plan.txt")
        
if __name__ == "__main__":
    main()