import streamlit as st
import random

# the ladder dictionary represents the question and scale for each question in the discovery process
ladder = {"General Wellbeing: On which step of the ladder would you say you personally feel you stand at this time?": list(range(0, 11)),
          "Positive Emotions: Where on the ladder do you stand now in terms of the amount of positive emotions in your life (e.g., experiencing happiness, joy, gratitude, etc.)?": list(range(0, 11)),
          "Engagement:  Where on the ladder do you stand now in terms of the amount of engagement in your life (e.g., being fully engaged in meaningful activities)?": list(range(0, 11)),
          "Positive relationships: Where on the ladder do you stand now in terms of the positive relationships in your life (e.g., connections with others that are authentic, supportive, and make you feel cared for/valued)?": list(range(0, 11)),
          "Meaning: Where on the ladder do you stand now in terms of the meaning in your life (e.g., being connected to and living your purpose in life)?": list(range(0, 11)),
          "Accomplishment: Where on the ladder do you stand now in terms of accomplishment in your life (e.g., pursuing and achieving goals, striving for greatness )?": list(range(0, 11)),
          "Physical: Where on the ladder do you stand now in terms of your physical health  (e.g., your overall physical well-being)?": list(range(0, 11)),
          "Mindset:  Where on the ladder do you stand now in terms of your mindset (e.g., a positive attitude towards yourself, growth, perseverance, hope and optimism)?": list(range(0, 11)),
          "Environment: Where on the ladder do you stand now in terms of the environments you live your life in (i.e. the surroundings you spend most of your time in)?": list(range(0, 11)),
          "Economic:  Where on the ladder do you stand now in terms of your economic security (e.g., your income, savings, spending, and investments)?": list(range(0, 11)),
         }

pillars = ["General Wellbeing", "Positive Emotions", "Engagement", "Positive Relationships", "Meaning", 
           "Accomplishment", "Physical", "Mindset", "Environment", "Economic"]

# initialize streamlit page and write ladder in a feedback form style. The range of the ladder is 0-10 and the user answers one question at a time with a slider
def main():
    st.title("Discovery")
    
    st.write("The Discovery process is a series of questions that will help you identify your current state of wellbeing. This will help you understand where you are now and where you want to be in the future. It will also help the AI narrow down the areas you want to focus on.")
    
    st.write("\nImagine a ladder where the top of the ladder represents the best possible life for you (10) and the bottom of the ladder represents the worst possible life for you (0)")
        
    # write questions and get slider input from user
    q1 = st.slider("General Wellbeing: On which step of the ladder would you say you personally feel you stand at this time?", min_value=0, max_value=10, value=5, step=1, key="General Wellbeing")
    q2 = st.slider("Positive Emotions: Where on the ladder do you stand now in terms of the amount of positive emotions in your life (e.g., experiencing happiness, joy, gratitude, etc.)?", min_value=0, max_value=10, value=5, step=1, key="Positive Emotions")
    q3 = st.slider("Engagement:  Where on the ladder do you stand now in terms of the amount of engagement in your life (e.g., being fully engaged in meaningful activities)?", min_value=0, max_value=10, value=5, step=1, key="Engagement")
    q4 = st.slider("Positive relationships: Where on the ladder do you stand now in terms of the positive relationships in your life (e.g., connections with others that are authentic, supportive, and make you feel cared for/valued)?", min_value=0, max_value=10, value=5, step=1, key="Positive Relationships")
    q5 = st.slider("Meaning: Where on the ladder do you stand now in terms of the meaning in your life (e.g., being connected to and living your purpose in life)?", min_value=0, max_value=10, value=5, step=1, key="Meaning")
    q6 = st.slider("Accomplishment: Where on the ladder do you stand now in terms of accomplishment in your life (e.g., pursuing and achieving goals, striving for greatness )?", min_value=0, max_value=10, value=5, step=1, key="Accomplishment")
    q7 = st.slider("Physical: Where on the ladder do you stand now in terms of your physical health  (e.g., your overall physical well-being)?", min_value=0, max_value=10, value=5, step=1, key="Physical")
    q8 = st.slider("Mindset:  Where on the ladder do you stand now in terms of your mindset (e.g., a positive attitude towards yourself, growth, perseverance, hope and optimism)?", min_value=0, max_value=10, value=5, step=1, key="Mindset")
    q9 = st.slider("Environment: Where on the ladder do you stand now in terms of the environments you live your life in (i.e. the surroundings you spend most of your time in)?", min_value=0, max_value=10, value=5, step=1, key="Environment")
    q10 = st.slider("Economic:  Where on the ladder do you stand now in terms of your economic security (e.g., your income, savings, spending, and investments)?", min_value=0, max_value=10, value=5, step=1, key="Economic")

    responses = {"General Wellbeing": q1, "Positive Emotions": q2, "Engagement": q3, "Positive Relationships": q4, 
                 "Meaning": q5, "Accomplishment": q6, "Physical": q7, "Mindset": q8, "Environment": q9, "Economic": q10}
        
    st.write("\nRanked Pillars:\n\n")
    
    for pillar in sorted(responses, key=responses.get, reverse=True):
        # print first three pillars highligthed in green, next three in dark yellow, and last three in red. Include the value of the pillar in the print statement
        st.markdown(f"<span style='color: {'#00c119' if responses[pillar] >= 7 else '#c6cd00' if responses[pillar] >= 4 else '#af024c'}'><b>{pillar}</b>: {responses[pillar]}</span>", unsafe_allow_html=True)
if __name__ == "__main__":
    main()