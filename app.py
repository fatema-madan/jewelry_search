import streamlit as st
import numpy as np
from PIL import Image
from sklearn.neighbors import NearestNeighbors

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.utils import img_to_array


st.set_page_config(
    page_title="Jewelry Visual Search",
    page_icon="💎",
    layout="wide"
)


@st.cache_resource
def load_model():
    model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        pooling="avg"
    )

    model.trainable = False

    return model


@st.cache_resource
def load_search_data():
    data = np.load(
        "jewelry_embeddings.npz",
        allow_pickle=True
    )

    embeddings = data["embeddings"]
    image_paths = data["image_paths"]

    nn = NearestNeighbors(
        n_neighbors=25,
        metric="cosine"
    )

    nn.fit(embeddings)

    return embeddings, image_paths, nn


model = load_model()

embeddings, image_paths, nn = load_search_data()

def extract_embedding(img):

    img = img.convert("RGB")
    img = img.resize((224, 224))

    img_array = img_to_array(img)

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    img_array = preprocess_input(img_array)

    embedding = model.predict(
        img_array,
        verbose=0
    )

    return embedding

st.title("💎 Jewelry Visual Search Engine")

st.write(
    "Upload a jewelry image or take a photo "
    "to find visually similar items."
)


option = st.radio(
    "Choose input method:",
    ["Upload Image", "Camera"]
)


uploaded_file = None

if option == "Upload Image":

    uploaded_file = st.file_uploader(
        "Upload jewelry image",
        type=["jpg", "jpeg", "png"]
    )

else:

    uploaded_file = st.camera_input(
        "Take a jewelry photo"
    )

if uploaded_file is not None:

    query_image = Image.open(uploaded_file)

    st.subheader("Query Image")

    st.image(
        query_image,
        width=300
    )


    query_embedding = extract_embedding(
        query_image
    )


    distances, indices = nn.kneighbors(
        query_embedding
    )


    st.subheader("Top 25 Similar Jewelry")


    cols = st.columns(5)

    for i, idx in enumerate(indices[0]):

        similarity = 1 - distances[0][i]

        with cols[i % 5]:

            image = Image.open(
                image_paths[idx]
            )

            st.image(
                image,
                use_container_width=True
            )

            st.write(
                f"Similarity: {similarity:.2f}"
            )
