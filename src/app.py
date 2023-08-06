import streamlit as st
import numpy as np
import random
import webcolors
from PIL import Image, ImageDraw
import plotly.express as px
import pandas as pd
import dithering_module
import matplotlib.pyplot as plt
import matplotlib.image as img
import stitch_pattern_maker
import color_palette as cp
from utils import set_bg
import ABC

import kMeans
from phq import *
from phq import kmeans_quantization, progressive_histogram_quantization

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def color_to_rgb(hex_code):
    hex_code = hex_code.lstrip('#')  # Remove '#' if present
    length = len(hex_code)

    # Convert hex code to RGB
    if length == 3:
        r = int(hex_code[0], 16) * 17
        g = int(hex_code[1], 16) * 17
        b = int(hex_code[2], 16) * 17
    elif length == 6:
        r = int(hex_code[0:2], 16)
        g = int(hex_code[2:4], 16)
        b = int(hex_code[4:6], 16)
    else:
        raise ValueError("Invalid hex code")

    return (r, g, b)

def rgb_to_hex(rgb):
    return '#%02x%02x%02x' % rgb

def main():

    # Web page initialization
    # Please set the path to the assests and style folders
    set_bg(r'...\assets\background.jpg')
    st.title("Image Quantization and Stitch Pattern Generator")

    local_css(r".../style/style.css")

    # Upload image
    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    selected_colors = None
    hex_colors = []  # Initialize hex_colors as an empty list
    save = None
    stitch_pattern = None
    kmeans = False
    phq = False
    abc = False
    quantize = None

    if uploaded_file is not None:

        col1, col2 = st.columns(2)
        with col1:
            st.image(uploaded_file, caption="Original Image", use_column_width=True)
        # Uploaded image is loaded as a numpy array
        image = Image.open(uploaded_file)
        image = np.array(image)

        with st.sidebar:
            # Three sliders to select the number of colors to be identified and the stitch size & width
            num_colors = st.slider("Number of Colors", min_value=2, max_value=100, value=5, step=1)
            stitch_size = st.slider("Stitch Size", min_value=2, max_value=100, value=25, step=1)
            stitch_width = st.slider("Stitch Width", min_value=2, max_value=20, value=10, step=1)
            st.write("---")
            dither = st.checkbox("Enable Dithering")
            st.write("---")

            algo = st.radio("Select the Quantization method", ("KMeans Quantization", "PHQ Quantization", "ABC Quantization"))
            if algo == "KMeans Quantization":
                kmeans = True
            if algo == "PHQ Quantization":
                phq = True
            if algo == "ABC Quantization":
                abc = True
            quantize = st.button("Quantize")
           
        # KMeans algorithm
        if quantize and kmeans:
            st.write("KMeans")
            st.info('The processing time depends on the size of the image. So, please wait until the result appear!', icon="ℹ️")
            quantized_image, colors = kMeans.kmeans_quantization(image, num_colors) # Quantization of the image
            centroid_colors = tuple(np.uint8(colors).tolist())

            # making the stitch pattern using a user given stitch size in pixels
            pattern = kMeans.create_pattern(quantized_image, stitch_size)
            original_reduced = kMeans.create_pattern(image, stitch_size)

            im_pil = Image.fromarray(quantized_image)
            im_pil.save("processed_image.jpg")

            psnr, ssim, mse = kMeans.calculate_metrics(original_reduced, pattern)

            col_psnr, col_ssim, col_mse = st.columns(3)

            with col_psnr:
                st.metric("PSNR", value="{:.2f}".format(psnr))
            with col_ssim:
                st.metric("SSIM", value="{:.2f}".format(ssim))
            with col_mse:
                st.metric("MSE", value="{:.2f}".format(mse))

            quantized_image = Image.open("processed_image.jpg")

            if dither:
                st.info('Image dithering takes some time and depends on the size of the image!', icon="ℹ️")
                q_image = np.array(quantized_image)
                quantized_image = dithering_module.floyd_steinberg_dithering(q_image) # Dithering the image using floyd steinberg method
                quantized_image = Image.fromarray(quantized_image)
            

            if quantized_image is not None:
                stitch_pattern = stitch_pattern_maker.stitch_pattern(quantized_image,stitch_size,stitch_width)
                stitch_pattern.save("stitch_pattern.png")
                st.image(stitch_pattern, caption="Stitch Pattern", use_column_width=True)            
    
            colors = colors.tolist()
            cp.show_rgb_values_box(colors)


        # PHQ algorithm
        if quantize and phq:
            st.write("PHQ")
            st.info('The processing time depends on the size of the image. So, please wait until the result appear!', icon="ℹ️")      
            quantized_histogram_r, quantized_histogram_g, quantized_histogram_b = progressive_histogram_quantization(image, desired_bins=5)
            quantized_image, colors = kmeans_quantization(image, quantized_histogram_r, quantized_histogram_g, quantized_histogram_b, n_clusters=5)
            # st.image(quantized_image)
            centroid_colors = tuple(np.uint8(colors).tolist())
            #print(centroid_colors)

            # making the stitch pattern using a user given stitch size in pixels
            pattern = kMeans.create_pattern(quantized_image, stitch_size)
            original_reduced = kMeans.create_pattern(image, stitch_size)

            # Display and save processed image

            im_pil = Image.fromarray(quantized_image)
            im_pil.save("processed_image.jpg")

            psnr, ssim, mse = kMeans.calculate_metrics(original_reduced, pattern)

            col_psnr, col_ssim, col_mse = st.columns(3)

            with col_psnr:
                st.metric("PSNR", value="{:.2f}".format(psnr))
            with col_ssim:
                st.metric("SSIM", value="{:.2f}".format(ssim))
            with col_mse:
                st.metric("MSE", value="{:.2f}".format(mse))

            quantized_image = Image.open("processed_image.jpg")

            if dither:
                st.info('Image dithering takes some time and depends on the size of the image!', icon="ℹ️")
                q_image = np.array(quantized_image)
                quantized_image = dithering_module.floyd_steinberg_dithering(q_image) # Dithering the image using floyd steinberg method
                quantized_image = Image.fromarray(quantized_image)
            

            if quantized_image is not None:
                stitch_pattern = stitch_pattern_maker.stitch_pattern(quantized_image,stitch_size,stitch_width)
                stitch_pattern.save("stitch_pattern.png")
                st.image(stitch_pattern, caption="Stitch Pattern", use_column_width=True)
                pdf_ready = True
          
            colors = colors.tolist()
            colors = [color[:3] for color in colors]
            cp.show_rgb_values_box(colors)
         

        # ABC algorithm
        if quantize and abc:
            st.write("ABC")
            st.info('The processing time depends on the size of the image. So, please wait until the result appear!', icon="ℹ️")
            quantized_image, colors = ABC.run_ABC(image, num_colors)  # calling the function
            centroid_colors = np.uint8(colors)

            # making the stitch pattern using a user given stitch size in pixels
            pattern = kMeans.create_pattern(quantized_image, stitch_size)
            original_reduced = kMeans.create_pattern(image, stitch_size)

            centroid_colors = centroid_colors.reshape((-1, centroid_colors.shape[-1]))

            # Display and save processed image

            im_pil = Image.fromarray(quantized_image)
            im_pil.save("processed_image.jpg")

            psnr, ssim, mse = kMeans.calculate_metrics(original_reduced, pattern)

            col_psnr, col_ssim, col_mse = st.columns(3)

            with col_psnr:
                st.metric("PSNR", value="{:.2f}".format(psnr))
            with col_ssim:
                st.metric("SSIM", value="{:.2f}".format(ssim))
            with col_mse:
                st.metric("MSE", value="{:.2f}".format(mse))

            quantized_image = Image.open("processed_image.jpg")

            if dither:
                st.info('Image dithering takes some time and depends on the size of the image!', icon="ℹ️")
                q_image = np.array(quantized_image)
                quantized_image = dithering_module.floyd_steinberg_dithering(q_image) # Dithering the image using floyd steinberg method
                quantized_image = Image.fromarray(quantized_image)
            

            if quantized_image is not None:
                stitch_pattern = stitch_pattern_maker.stitch_pattern(quantized_image,stitch_size,stitch_width)
                stitch_pattern.save("stitch_pattern.png")
                st.image(stitch_pattern, caption="Stitch Pattern", use_column_width=True)
            
            colors = colors.reshape((-1, colors.shape[-1]))
            colors = colors.tolist()
            print(colors)
            cp.show_rgb_values_box(colors)
    
        with st.sidebar:
            st.write("---")
            st.subheader("Save and Download")
            save = st.button("Save as PDF")
            if save:
                fig = plt.figure(figsize=(6, 6))
                ax = fig.add_axes([0, 0, 1, 1], xticks=[], yticks=[], frame_on=False)
                im = Image.open("stitch_pattern.png")
                plt.imshow(im)
                plt.savefig("stitch_pattern.pdf")
                print("pdf done")
                download_ready =  True

                with open("stitch_pattern.pdf", 'rb') as f:
                    pdf_bytes = f.read()
            
                st.download_button("Download PDF", data=pdf_bytes, file_name="stitch_pattern.pdf", mime="application/pdf")
    if quantize:
        st.write("---")
        if st.button("Reset"):
            uploaded_file = None
      



if __name__ == "__main__":
    main()
