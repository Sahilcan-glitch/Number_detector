import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf

st.title('Handwritten Digit Recognizer')

# Load the trained model
model = tf.keras.models.load_model('mnist_model.keras')

# Create a canvas component
canvas_result = st_canvas(
    fill_color='black',
    stroke_width=25,
    stroke_color='white',
    background_color='black',
    width=300,
    height=300,
    drawing_mode='freedraw',
    key='canvas'
)

if st.button('Predict'):
    if canvas_result.image_data is not None:
        # Preprocess the image
        img = Image.fromarray(np.uint8(canvas_result.image_data)).convert('L')
        img = img.resize((28, 28))
        img = ImageOps.invert(img)
        img = np.array(img) / 255.0
        img = img.reshape(1, 28, 28, 1)

        # Display the processed image
        st.image(img.reshape(28, 28), caption='Processed Image', width=150)

        # Predict the digit
        prediction = model.predict(img)
        predicted_digit = np.argmax(prediction)
        st.write(f'**Predicted Digit:** {predicted_digit}')

        # Display prediction probabilities
        st.bar_chart(prediction[0])

        # Print raw prediction probabilities for debugging
        st.write('Prediction Probabilities:', prediction[0])
    else:
        st.write("Please draw a digit on the canvas.")
