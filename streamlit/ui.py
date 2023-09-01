import io

import requests
from PIL import Image
from requests_toolbelt.multipart.encoder import MultipartEncoder

import streamlit as st

# interact with FastAPI endpoint
backend = "http://fastapi:8000/segmentation"


def process(image, server_url: str):
    encoded_data = MultipartEncoder(fields={"file": ("filename", image, "image/jpeg")})

    request = requests.post(
        server_url,
        data=encoded_data,
        headers={"Content-Type": encoded_data.content_type},
        timeout=8000,
    )

    return request


# construct UI layout
st.title("Semantic Image Segmentation Demo")

st.write(
    """Image Semantic Segmentation using models implemented in PyTorch and converted for inference.
        This streamlit uses a FastAPI service as backend.
        Visit this URL at `:8000/docs` for FastAPI documentation."""
)  # Description and Instructions

input_image = st.file_uploader("Insert or Browse Image")  # Image Upload Widget

if st.button("Get Segmentation Mask"):
    col1, col2 = st.columns(2)

    if input_image:
        segments = process(input_image, backend)
        original_image = Image.open(input_image).convert("RGB")
        segmented_image = Image.open(io.BytesIO(segments.content)).convert("RGB")
        col1.header("Image Original")
        col1.image(original_image, use_column_width=True)
        col2.header("Segmentation Mask")
        col2.image(segmented_image, use_column_width=True)

    else:
        # handle case with no image
        st.write("Insert an image!")
