%%writefile app.py

import streamlit as st
import numpy as np

from PIL import Image
from sklearn.neighbors import NearestNeighbors

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.utils import img_to_array


# -----------------------------------
# App settings
# -----------------------------------

st.set_page_config(
    page_title="Jewelry Visual Search",
    page_icon="💎",
    layout="wide"
)

TOP_K = 25


# -----------------------------------
# Load MobileNetV2
# -----------------------------------

@st.cache_resource
def load_model():

    model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        pooling="avg"
    )

    model.trainable = False

    return model


# -----------------------------------
# Load saved embeddings
# -----------------------------------

@st.cache_resource
def load_search_index():

    data = np.load(
        "jewelry_embeddings.npz",
        allow_pickle=True
    )

    embeddings = data["embeddings"]
    image_paths = data["image_paths"]

    nn = NearestNeighbors(
        n_neighbors=TOP_K,
        metric="cosine"
    )

    nn.fit(embeddings)

    return embeddings, image_paths, nn


# -----------------------------------
# Extract embedding
# -----------------------------------

def extract_embedding(model, img):

    img = img.convert("RGB")
    img = img.resize((224, 224))

    img_array = img_to_array(img)

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    img_array = preprocess_input(
        img_array
    )

    embedding = model.predict(
        img_array,
        verbose=0
    )

    return embedding


# -----------------------------------
# Load model and search index
# -----------------------------------

model = load_model()

embeddings, image_paths, nn = load_search_index()


# -----------------------------------
# UI
# -----------------------------------

st.title("💎 Jewelry Visual Search Engine")

st.write(
    "Upload a jewelry image or take a photo "
    "to find visually similar jewelry."
)


source = st.radio(
    "Choose image source:",
    ["Upload Image", "Take a Photo"]
)


query_image = None


# Upload image
if source == "Upload Image":

    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        query_image = Image.open(
            uploaded_file
        )


# Camera
else:

    camera_file = st.camera_input(
        "Take a photo"
    )

    if camera_file is not None:

        query_image = Image.open(
            camera_file
        )


# -----------------------------------
# Search
# -----------------------------------

if query_image is not None:

    st.subheader("Query Image")

    st.image(
        query_image,
        width=300
    )


    # Similarity threshold
    threshold = st.slider(
        "Minimum similarity",
        min_value=0.0,
        max_value=1.0,
        value=0.60,
        step=0.05
    )


    # Extract query embedding
    query_embedding = extract_embedding(
        model,
        query_image
    )


    # Search nearest images
    distances, indices = nn.kneighbors(
        query_embedding
    )


    matches = []

    for distance, idx in zip(
        distances[0],
        indices[0]
    ):

        similarity = 1 - distance

        if similarity >= threshold:

            matches.append(
                (
                    image_paths[idx],
                    similarity
                )
            )


    # -----------------------------------
    # Results
    # -----------------------------------

    if len(matches) == 0:

        st.warning(
            "No similar jewelry found. "
            "Try another image or lower the threshold."
        )


    else:

        st.subheader(
            f"Top {len(matches)} Similar Jewelry"
        )

        # 5 images per row
        for row_start in range(
            0,
            len(matches),
            5
        ):

            cols = st.columns(5)

            row_matches = matches[
                row_start:row_start + 5
            ]

            for col, (
                image_path,
                similarity
            ) in zip(
                cols,
                row_matches
            ):

                with col:

                    result_image = Image.open(
                        image_path
                    )

                    st.image(
                        result_image,
                        use_container_width=True
                    )

                    st.caption(
                        f"Similarity: "
                        f"{similarity * 100:.2f}%"
                    )
