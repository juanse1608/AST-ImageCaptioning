# Import libreries
import streamlit as st
from predictions_type_0 import evaluate, prob_evaluate
import session_state
import json

# Run the predict page
st.title('Machine Learning Web App - Image Captioning')
st.header("Final Project - Advanced Statistics Topics: ML and DS")
st.write('''Upload a photo and see the predicted caption for it''')

# File uploader allows user to add their own image
uploaded_file = st.file_uploader(label="Upload Image", type=["png", "jpeg", "jpg"])
# Setup session state to remember state of app so refresh isn't always needed
# See: https://discuss.streamlit.io/t/the-button-inside-a-button-seems-to-reset-the-whole-app-why/1051/11 
session_state = session_state.get(pred_button=False)

# Create logic for app flow
if not uploaded_file:
    st.warning("Please upload an image.")
    st.stop()
else:
    session_state.uploaded_image = uploaded_file.read()
    st.image(session_state.uploaded_image, use_column_width=True)
    pred_button = st.button("Predict (Deterministic)")
    prob_pred_button = st.button("Predict (Random)")

# Did the user press the predict button?
if pred_button:
    session_state.pred_button = True 
# Did the user press the predict button?
if prob_pred_button:
    session_state.prob_pred_button = True 

# And if they did...
if session_state.pred_button:
    session_state.pred_caption, session_state.attention_plot = evaluate(session_state.uploaded_image)
    st.write(f"Prediction: {session_state.pred_caption}")

    # Create feedback mechanism (building a data flywheel)
    session_state.feedback = st.selectbox(
        "Is this correct?",
        ("Select and option", "Yes", "Kinda", "No"), )
    if session_state.feedback == "Select an option":
        pass
    elif session_state.feedback == "Yes":
        st.write("Thank you for your feedback!")
        # Log prediction information to terminal (this could be stored in Big Query or something like that)
        # print(update_logger(image=session_state.image,
        #                    model_used=MODEL,
        #                    pred_class=session_state.pred_class,
        #                    pred_conf=session_state.pred_conf,
        #                    correct=True))
    elif session_state.feedback == "No":
        session_state.correct_class = st.text_input("What should the correct caption be?")
        if session_state.correct_class:
            st.write("Thank you for that, we'll use your help to make our model better!")
            # Log prediction information to terminal (this could be stored in Big Query or something like that)
            #print(update_logger(image=session_state.image,
            #                    model_used=MODEL,
            #                    pred_class=session_state.pred_class,
            #                    pred_conf=session_state.pred_conf,
            #                    correct=False,
            #                    user_label=session_state.correct_class))
    elif session_state.feedback == "Kinda":
        session_state.correct_class = st.text_input("What should a better caption be?")
        if session_state.correct_class:
            st.write("Thank you for that, we'll use your help to make our model better!")
            # Log prediction information to terminal (this could be stored in Big Query or something like that)
            #print(update_logger(image=session_state.image,
            #                    model_used=MODEL,
            #                    pred_class=session_state.pred_class,
            #                    pred_conf=session_state.pred_conf,
            #                    correct=False,
            #                    user_label=session_state.correct_class))
if session_state.prob_pred_button:
    session_state.prob_pred_caption, session_state.prob_attention_plot = prob_evaluate(session_state.uploaded_image)
    st.write(f"Prediction: {session_state.prob_pred_caption}")

    # Create feedback mechanism (building a data flywheel)
    session_state.feedback = st.selectbox(
        "Is this correct?",
        ("Select and option", "Yes", "Kinda", "No"), )
    if session_state.feedback == "Select an option":
        pass
    elif session_state.feedback == "Yes":
        st.write("Thank you for your feedback!")
        # Log prediction information to terminal (this could be stored in Big Query or something like that)
        # print(update_logger(image=session_state.image,
        #                    model_used=MODEL,
        #                    pred_class=session_state.pred_class,
        #                    pred_conf=session_state.pred_conf,
        #                    correct=True))
    elif session_state.feedback == "No":
        session_state.correct_class = st.text_input("What should the correct caption be?")
        if session_state.correct_class:
            st.write("Thank you for that, we'll use your help to make our model better!")
            # Log prediction information to terminal (this could be stored in Big Query or something like that)
            #print(update_logger(image=session_state.image,
            #                    model_used=MODEL,
            #                    pred_class=session_state.pred_class,
            #                    pred_conf=session_state.pred_conf,
            #                    correct=False,
            #                    user_label=session_state.correct_class))
    elif session_state.feedback == "Kinda":
        session_state.correct_class = st.text_input("What should a better caption be?")
        if session_state.correct_class:
            st.write("Thank you for that, we'll use your help to make our model better!")
            # Log prediction information to terminal (this could be stored in Big Query or something like that)
            #print(update_logger(image=session_state.image,
            #                    model_used=MODEL,
            #                    pred_class=session_state.pred_class,
            #                    pred_conf=session_state.pred_conf,
            #                    correct=False,
            #                    user_label=session_state.correct_class))