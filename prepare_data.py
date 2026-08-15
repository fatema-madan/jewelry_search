import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
import keras
from PIL import Image
from sklearn.neighbors import NearestNeighbors

warnings.filterwarnings("ignore")

import glob

path = "/kaggle/input/datasets/sapnilpatel/tanishq-jewellery-dataset/Jewellery_Data/"

image_paths = glob.glob(path + "*/*.jpg")

image_paths

len(image_paths)

from tensorflow.keras.applications import MobileNetV2

backbone = MobileNetV2(
    weights="imagenet",
    include_top=False,
    pooling="avg"
)

backbone.trainable = False
backbone.summary()

from tensorflow.keras.utils import load_img

load_img(image_paths[0])

from tensorflow.keras.utils import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

img = load_img(
    image_paths[0],
    target_size=(224, 224)
)

img_array = img_to_array(img)

img_array = np.expand_dims(
    img_array,
    axis=0
)

img_array = preprocess_input(img_array)

img_array.shape

embedding = backbone.predict(img_array)

embedding.shape

def extract_embedding(image_path):
    img = load_img(
        image_path,
        target_size=(224, 224)
    )

    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    embedding = backbone.predict(
        img_array,
        verbose=0
    )

    return embedding[0]

embeddings = []

for image_path in image_paths:
    embedding = extract_embedding(image_path)
    embeddings.append(embedding)

embeddings = np.array(embeddings)

embeddings.shape

np.savez(
    "jewelry_embeddings.npz",
    embeddings=embeddings,
    image_paths=np.array(image_paths)
)

data = np.load("jewelry_embeddings.npz")

print(data["embeddings"].shape)
print(len(data["image_paths"]))

nn = NearestNeighbors(
    n_neighbors=5,
    metric="cosine"
)

nn.fit(embeddings)

query_index = 0

query_embedding = embeddings[query_index].reshape(1, -1)

distances, indices = nn.kneighbors(query_embedding)

print("Indices:", indices)
print("Distances:", distances)

plt.figure(figsize=(15, 4))

# Query image
plt.subplot(1, 6, 1)

query_img = Image.open(image_paths[query_index])

plt.imshow(query_img)
plt.title("Query")
plt.axis("off")


# Similar images
for i, idx in enumerate(indices[0]):
    plt.subplot(1, 6, i + 2)

    similar_img = Image.open(image_paths[idx])

    plt.imshow(similar_img)
    plt.title(f"Match {i+1}")
    plt.axis("off")

plt.tight_layout()
plt.show()

