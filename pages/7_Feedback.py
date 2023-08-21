import streamlit as st
from utils import send_email

def main():
    st.title("Group Discussion")
    st.markdown("""Greetings and expectations for group discussion""")
    
    st.markdown("""instructions for group discussion""")
    
    user_name = st.text_input("Name")
    user_email = st.text_input("Email")
    
    q1 = st.text_area("1. How was the experience overall?")
    q2 = st.text_area("2. What did you like about it?")
    q3 = st.text_area("3. What did you not like about it?")
    q4 = st.text_area("4. What features would you like to see added?")
    
    q5 = st.slider("5. How likely are you to recommend something like this to a friend?", 1, 7)
    q6 = st.slider("6. How helpful was the experience in assessing your standing on the selected PERMA pillar?", 1, 7)
    q7 = st.slider("7. How helpful was the experience in establishing actionable steps towards improving the selected PERMA pillar?", 1, 7)
    q8 = st.slider("8. How manageable is the action plan you developed?", 1, 7)
    q9 = st.slider("9. How likely are you to follow through on the action plan you developed?", 1, 7)
    
    q10 = st.text_area("10. What kind of changes to your action plan would make you more likely to follow through on the action plan you developed?")
    q11 = st.text_area("11. What kind of support from the platform would make you more likely to follow through on the action plan you developed?")

    # merge all feedback into one string well formatted report
    responses = f"""1. How was the experience overall? \n{q1}\n
                    2. What did you like about it? \n{q2}\n    
                    3. What did you not like about it? \n{q3}\n
                    4. What features would you like to see added? \n{q4}\n
                    5. How likely are you to recommend something like this to a friend? \n{q5}\n
                    6. How helpful was the experience in assessing your standing on the selected PERMA pillar? \n{q6}\n
                    7. How helpful was the experience in establishing actionable steps towards improving the selected PERMA pillar? \n{q7}\n
                    8. How manageable is the action plan you developed? \n{q8}\n
                    9. How likely are you to follow through on the action plan you developed? \n{q9}\n
                    10. What kind of changes to your action plan would make you more likely to follow through on the action plan you developed? \n{q10}\n
                    11. What kind of support from the platform would make you more likely to follow through on the action plan you developed? \n{q11}\n
                    
                    The following information was provided by the user:
                    Name: {user_name}
                    Email: {user_email}
                    Date: {dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
                    """

    if st.button("Submit Feedback"):
        send_email(user_name, user_email, responses)
        st.balloons()
        st.success("Your responses has been submitted successfully! We appreciate the time you took to help us improve Atlas Intelligence.")


if __name__ == "__main__":
    main()