import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
import streamlit as st
import numpy as np
import json
import matplotlib.pyplot as plt

# set the width of the web page
st.set_page_config(page_title="My Streamlit App", page_icon=":earth_asia:")

# display the header
st.header(':blue[Check Food Allergy in just one click] :point_up_2:')
st.write(':red[Note: Please insert one food image at a time to receive accurate information.]')

# load the model
model = load_model('C:/Users/NITIN SAINI/image_classifier/Allergy_Detection.keras')
data_cat = ['Apple','Coconut','Fish','Grapes','Kiwi','Mango','Orange',
            'Papaya','banana','rice','tea','tomato']

with open('allergy_data.json') as f:
    allergy_data = json.load(f)

# set the height and width of the image
img_height = 180
img_width = 180

# Add dropdown menu
selected_class = st.selectbox("Select type of the image:", options=['Choose category'] + data_cat)

# Check if a category has been selected before allowing image upload
if selected_class != 'Choose category':
    uploaded_file = st.file_uploader("Choose an image...", type="jpg")
    
    # Check if an image has been uploaded
    if uploaded_file is not None:
        image_load = tf.keras.utils.load_img(uploaded_file, target_size=(img_height, img_width))
        
        img_arr = tf.keras.utils.array_to_img(image_load)
        img_bat = tf.expand_dims(img_arr,0)

        predict = model.predict(img_bat)

        score = tf.nn.softmax(predict)

        food_name = selected_class

        # Check if the selected category and the detected class are the same
        if food_name.lower() == data_cat[np.argmax(score)].lower():
            st.image(uploaded_file,width=300)

            st.write('food in image is ' + food_name + ' with accuracy of ' + '{:.6f}'.format(np.max(score)*100) + '%')

            for food in allergy_data['food_allergies']:
                if food['name'].lower() == food_name.lower():
                    st.subheader(':green[Allergy Information]',divider='rainbow')

                    st.write('<p style="font-size:25px"><span style="color:red"><b>Description:-</b></span> ' + food['description'] + '</p>', unsafe_allow_html=True)

                    st.write('<p style="font-size:25px"><span style="color:red"><b>Common Symptoms:-</b></span> ' + ', '.join(food['common_symptoms']) + '</p>', unsafe_allow_html=True)

                    st.write('<p style="font-size:25px"><span style="color:red"><b>Avoidance Tips:-</b></span> ' + ', '.join(food['avoidance_tips']) + '</p>', unsafe_allow_html=True)

                    # Plotting the Symptoms vs Symptoms values graph
                    Symptoms_values = list(food['Symptoms'].values())
                    Symptoms_values = [(val / 60) * 100 for val in Symptoms_values]
                    Symptoms_names = list(food['Symptoms'].keys())
                    fig = plt.figure(figsize=(6, 2))
                    plt.bar(Symptoms_names, Symptoms_values)
                    plt.xticks(fontsize=5)
                    plt.yticks(fontsize=5)
                    plt.xlabel('Symptoms', fontsize=10)
                    plt.ylabel('Chances of Risk', fontsize=10)
                    st.pyplot(fig)
        else:
            st.write(':red[There are some error occurred with this image. Please try another image.]')
    else:
        st.write('Please upload an image.')
