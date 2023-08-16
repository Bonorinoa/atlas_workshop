import streamlit as st
import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

def send_email(user_name, user_email, feedback):
    smtp_port = 587                 # Standard secure SMTP port
    smtp_server = "smtp.gmail.com"  # Google SMTP Server

    email_from = "agbonorino21@gmail.com" 
    email_to = "atlas.intelligence21@gmail.com" 

    pswd = "arqxfjbvwuyybopi"       # App password for gmail
    
    message = f"""Feedback from {user_name} ({user_email}):

    {feedback}
    """

    ### SEND EMAIL ###
    simple_email_context = ssl.create_default_context()

    try:
        # Connect to the server
        print("Connecting to server...")
        TIE_server = smtplib.SMTP(smtp_server, smtp_port)
        TIE_server.starttls(context=simple_email_context)
        TIE_server.login(email_from, pswd)
        print("Connected to server :-)")
        
        # Create the MIMEText object with the message and encoding
        msg = MIMEMultipart()
        msg.attach(MIMEText(message, "plain", "utf-8"))
        msg["From"] = email_from
        msg["To"] = email_to
        msg["Subject"] = "Feedback Form - Atlas Demo ({})".format(user_name)
        
        # Send the actual email
        print()
        print(f"Sending email to - {email_to}")
        TIE_server.send_message(msg)
        print(f"Email successfully sent to - {email_to}")

    # If there's an error, print it out
    except Exception as e:
        print(e)

    # Close the port
    finally:
        TIE_server.quit()
    
    

def main():
    st.title("User Feedback Form")
    st.markdown("""Greetings from the Atlas Intelligence Team,
        We greatly appreciate your time in completing our initial wellness assessment. Please see your wellness report and suggested wellness goals/activities in the text file attached. 
        The report is used internally by the AI to suggest the goals and activities. In practice, this report will most likely not be shared with users as its only purpose is to capture the essence of the responses in the survey. 
        We apologize for the raw format, we will work on these details as we progress with this project.""")
    
    st.markdown("""After taking the time to thoroughly examine these materials, we would greatly appreciate your time in sharing your thoughts on the model's suggestions (report, goals, and activities). 
        Here are a few questions that will really help us start tuning our AI to the purposes of real humans seeking to improve their wellbeing. Simply reply to this email with the feedback if you wish to do so :)""")
    
    user_name = st.text_input("Name")
    user_email = st.text_input("Email")
    q1 = st.text_area("1. Did your wellness report make sense given your responses? If not, please explain why")
    q2 = st.text_area("2. Do you think your wellness report accurately describes the general state of your wellbeing? If not, please explain why")
    q3 = st.text_area("3. Did you find the generated goals and suggestions useful? Please explain.")
    q4 = st.text_area("4. Do you feel like the generated resources might be helpful in improving your wellbeing? Please explain.")
    q5 = st.text_area("5. Would you recommend a service like this to a friend? Please explain")
    q6 = st.text_area("6. What can we do to improve your overall experience with this service or help make this service more useful?")
    
    extra = st.text_area("Any additional comments or suggestions?", max_chars=1000)
    
    # merge all feedback into one string well formatted
    feedback = f"""1. Did your wellness report make sense given your responses? If not, please explain why. \n{q1}\n
                   2. Do you think your wellness report accurately describes the general state of your wellbeing? If not, please explain why. \n{q2}\n
                   3. Did you find the generated goals and suggestions useful? Please explain. \n{q3}\n
                   4. Do you feel like the generated resources might be helpful in improving your wellbeing? Please explain. \n{q4}\n
                   5. Would you recommend a service like this to a friend? Please explain. \n{q5}\n
                   6. What can we do to improve your overall experience with this service or help make this service more useful? \n{q6}\n
                   
                   Additional comments: \n{extra}. 
    """

    if st.button("Submit Feedback"):
        send_email(user_name, user_email, feedback)
        st.balloons()
        st.success("Your feedback has been submitted successfully! We appreciate the time you took to help us improve Atlas Intelligence.")

if __name__ == "__main__":
    main()