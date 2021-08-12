# Import libreries
import streamlit as st
from streamlit.server.server import SessionInfo
from predictions_type_0 import evaluate, prob_evaluate
import session_state
import json
import pandas as pd
import PIL.Image as Image
import session_info
import io
import os
from google.cloud import storage, bigquery 

# Feedback type
cloud_feedback = True
try:
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'GOOGLE_ACC_CREDENTIALS.json'
    storage_cl = storage.Client()
    bigq_cl = bigquery.Client()
except:
    creds = st.secrets["GOOGLE_ACC_CREDENTIALS"]
    storage_cl = storage.Client(credentials=creds)
    bigq_cl = bigquery.Client(credentials=creds)


bucket_name = 'img-captioning-feedback'

# Run the predict page
st.title('Machine Learning Web App - Image Captioning')
st.header("Final Project - Advanced Statistics Topics: ML and DS")
st.header("Click [here](https://github.com/juanse1608/AST-ImageCaptioning/blob/main/README.md) to know more about the project!")
st.write('''Upload a photo and see the predicted caption for it''')

# File uploader allows user to add their own image
uploaded_file = st.file_uploader(label="Upload Image", type=["png", "jpeg", "jpg"])
# Setup session state to remember state of app so refresh isn't always needed
# See: https://discuss.streamlit.io/t/the-button-inside-a-button-seems-to-reset-the-whole-app-why/1051/11 
session_state = session_state.get(pred_button=False)

# Create logic for app flow
if not uploaded_file:
    st.warning("Please upload an image.")
    session_state.pred_button = False
    st.stop()
else:
    session_state.uploaded_image = uploaded_file.read()
    st.image(session_state.uploaded_image, use_column_width=True)
    value = st.selectbox("Select Prediction Type", ("Argmax", "Random"), help='''__Argmax__ picks the value/token with the highest probability.
    __Random__ picks the value/token randomly using the distribution of the predictions.''')

if value == 'Argmax':
    # pred_button = st.button("Predict")
    # Did the user press the predict button?

    #if pred_button:
    session_state.pred_button = True
    
    # And if they did...
    if session_state.pred_button:
        session_state.pred_caption, session_state.attention_plot = evaluate(session_state.uploaded_image)
        val = session_state.pred_caption
        st.write(f"The prediction via argmax probability for this caption is: __{val}.__")
    # Create feedback mechanism (building a data flywheel)
    session_state.feedback = st.selectbox("Is this correct?", ("Select and option", "Yes", "No"))
    if cloud_feedback:
        if session_state.feedback == "Select an option":
            pass
        elif session_state.feedback == "Yes":
            feedback_bucket = storage_cl.get_bucket(bucket_name)
            blobs = storage_cl.list_blobs(bucket_name, prefix='images/', delimiter='/')
            try: 
                files = []
                for f in blobs:
                    f_n = f.name.split('/')[1]
                    try:
                        files.append(int(f_n.split('.')[0]))
                    except:
                        pass
                new_file = str(max(files)+1) + '.jpg'
            except:
                new_file = '0.jpg'
            img = Image.open(io.BytesIO(session_state.uploaded_image))
            img.save(f'{new_file}')
            blob = feedback_bucket.blob(f'images/{new_file}')
            blob.upload_from_filename(new_file)
            os.remove(f'{new_file}')
            new_caption = pd.DataFrame({'PATH': [new_file], 'CAPTION': val})
            st.write("Thank you for your feedback!")
        elif session_state.feedback == "No":
            session_state.correct_class = st.text_input("What should the correct caption be?")
            if session_state.correct_class:
                try: 
                    files = os.listdir(os.getcwd() + '/Data/Feedback/Images/')
                    files = [f for f in files if not f.startswith('.')]
                    files = [int(f.split('.')[0]) for f in files]
                    new_file = str(max(files)+1) + '.jpg'
                except:
                    new_file = '0.jpg'
                img = Image.open(io.BytesIO(session_state.uploaded_image))
                img.save(f'Data/Feedback/Images/{new_file}')
                new_caption = pd.DataFrame({'PATH': [new_file], 'CAPTION': session_state.correct_class})
                try:
                    caps = pd.read_csv('Data/Feedback/Captions/captions.csv')
                    caps = pd.concat([caps, new_caption], axis=0).reset_index(drop=True)
                    caps.to_csv('Data/Feedback/Captions/captions.csv', index=False)
                except:
                    new_caption.to_csv('Data/Feedback/Captions/captions.csv', index=False)
                st.write("Thank you for that, we'll use your help to make our model better!")
                # Log prediction information to terminal (this could be stored in Big Query or something like that)
        
        

    else:
        if session_state.feedback == "Select an option":
            pass
        elif session_state.feedback == "Yes":
            try: 
                files = os.listdir(os.getcwd() + '/Data/Feedback/Images/')
                files = [f for f in files if not f.startswith('.')]
                files = [int(f.split('.')[0]) for f in files]
                new_file = str(max(files)+1) + '.jpg'
            except:
                new_file = '0.jpg'
            img = Image.open(io.BytesIO(session_state.uploaded_image))
            img.save(f'Data/Feedback/Images/{new_file}')
            new_caption = pd.DataFrame({'PATH': [new_file], 'CAPTION': val})
            try:
                caps = pd.read_csv('Data/Feedback/Captions/captions.csv')
                caps = pd.concat([caps, new_caption], axis=0).reset_index(drop=True)
                caps.to_csv('Data/Feedback/Captions/captions.csv', index=False)
            except:
                new_caption.to_csv('Data/Feedback/Captions/captions.csv', index=False)
            st.write("Thank you for your feedback!")
        elif session_state.feedback == "No":
            session_state.correct_class = st.text_input("What should the correct caption be?")
            if session_state.correct_class:
                try: 
                    files = os.listdir(os.getcwd() + '/Data/Feedback/Images/')
                    files = [f for f in files if not f.startswith('.')]
                    files = [int(f.split('.')[0]) for f in files]
                    new_file = str(max(files)+1) + '.jpg'
                except:
                    new_file = '0.jpg'
                img = Image.open(io.BytesIO(session_state.uploaded_image))
                img.save(f'Data/Feedback/Images/{new_file}')
                new_caption = pd.DataFrame({'PATH': [new_file], 'CAPTION': session_state.correct_class})
                try:
                    caps = pd.read_csv('Data/Feedback/Captions/captions.csv')
                    caps = pd.concat([caps, new_caption], axis=0).reset_index(drop=True)
                    caps.to_csv('Data/Feedback/Captions/captions.csv', index=False)
                except:
                    new_caption.to_csv('Data/Feedback/Captions/captions.csv', index=False)
                st.write("Thank you for that, we'll use your help to make our model better!")
                # Log prediction information to terminal (this could be stored in Big Query or something like that)

            
       
elif value == 'Random':
    session_state.pred_button = False
    pred_button_new = st.button("Predict")
    # Did the user press the predict button?

    if pred_button_new:
        session_state.pred_button = True
    # And if they did...
    if session_state.pred_button:
        session_state.pred_caption, session_state.attention_plot = prob_evaluate(session_state.uploaded_image)
        val = session_state.pred_caption
        st.write(f"The prediction via random selection for this caption is: __{val}.__")
else:
    st.write('There was some error in prediction type!')
