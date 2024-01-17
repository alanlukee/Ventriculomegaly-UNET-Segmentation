import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import os

IMAGE_HEIGHT = 256  
IMAGE_WIDTH = 256  


model = load_model("unet_ventricle_segmentation_model.h5")
# st.write("Model Loaded Successfully!")


def get_corresponding_mask_path(image_path):
    image_name = os.path.basename(image_path)
    mask_name = f"mask{os.path.splitext(image_name)[0][5:]}.png"
    mask_path = os.path.join("/Users/alanluke/Desktop/W_masks", mask_name)  
    return mask_path


def main():
    st.title("Ventricle Segmentation ")

    st.sidebar.header("Input Parameters")


    uploaded_image = st.sidebar.file_uploader("Upload MRI Image", type=["jpg", "jpeg", "png"])

  
    if uploaded_image is not None:

     
        col1, col2 = st.columns(2)
        col1.subheader("MRI Image")
        col1.image(uploaded_image, caption="MRI Image", width=300)
        corresponding_mask_path = get_corresponding_mask_path(uploaded_image.name)
        
        if os.path.exists(corresponding_mask_path):
            col2.subheader("Ground Truth Mask")
            col2.image(corresponding_mask_path, caption="Ground Truth Mask", width=300, channels="GRAY")

            predict_button = st.sidebar.button("Predict")

            if predict_button:
                with st.spinner("Predicting..."):
                   
                    predicted_mask = predict_mask(uploaded_image)

                display_results(uploaded_image, corresponding_mask_path, predicted_mask)
        else:
            st.warning("Corresponding mask not found for the selected MRI image.")
            
def predict_mask(image):
    new_img = tf.keras.preprocessing.image.load_img(image, target_size=(IMAGE_HEIGHT, IMAGE_WIDTH))
    new_img_array = tf.keras.preprocessing.image.img_to_array(new_img)
    new_img_array = np.expand_dims(new_img_array, axis=0)  
    predicted_mask = model.predict(new_img_array)
    threshold = 0.1  
    binary_mask = (predicted_mask > threshold).astype(np.uint8)[0, :, :, 0]
    return binary_mask


def display_results(image_path, ground_truth_mask_path, predicted_mask):
    new_img = tf.keras.preprocessing.image.load_img(image_path, target_size=(IMAGE_HEIGHT, IMAGE_WIDTH))
    predicted_mask_display = (predicted_mask * 255).astype(np.uint8)

    st.subheader("Predicted Mask")
    
    st.image(predicted_mask_display, caption="Predicted Mask", width=600, channels="GRAY")

if __name__ == "__main__":
    main()
