import streamlit as st
import webcolors
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
import plotly.graph_objects as go

dmc_colors = webcolors.CSS3_NAMES_TO_HEX

def hex_to_rgb(hex_code):
    rgb = webcolors.hex_to_rgb(hex_code)
    return rgb

dmc_col = []
for col in dmc_colors.values():
  dmc_col.append(hex_to_rgb(col))


def match_with_dmc(dmc_col, rgb):
    r, g, b = rgb
    dmc = np.array(dmc_col)
    distances = np.sqrt(np.sum((dmc - [r, g, b]) ** 2, axis=1))
    closest = np.min(distances)
    closest_index = np.where(distances == closest)
    return dmc[closest_index[0]].tolist()

def show_rgb_values_box(colors):

    matched = []
    for i in range(len(colors)):
        list_item = (match_with_dmc(dmc_col, colors[i]))
        matched.append(list_item)
    
    st.subheader("Matched Colors")
    print(matched)
    for rgb in matched:
        color = f"rgb({rgb[0][0]}, {rgb[0][1]}, {rgb[0][2]})"
        st.markdown(f'<div style="width: 50px; height: 50px; background-color: {color};"></div>', unsafe_allow_html=True)