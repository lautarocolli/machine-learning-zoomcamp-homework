
import numpy as np
import onnxruntime as ort

from io import BytesIO
from urllib import request
from PIL import Image

target_size = (200,200)

def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img


def prepare_image(img, target_size):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    return img


def preprocess_image(url):
    img = download_image(url)
    img = prepare_image(img,target_size)

    # Convert to numpy and scale to [0,1]
    img = np.array(img).astype(np.float32) / 255.0

    # HWC → CHW
    img = np.transpose(img, (2, 0, 1))

    # Normalize (ImageNet stats)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3,1,1)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3,1,1)

    img = (img - mean) / std

    # Add batch dimension
    img = np.expand_dims(img, axis=0)

    return img


def predict(img_array):
    session = ort.InferenceSession("hair_classifier_empty.onnx")
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    session_run = session.run(
        [output_name],
        {input_name: img_array}
    )

    prediction = session_run[0][0].tolist()

    return prediction


def lambda_handler(event, context):
    url = event['url']
    img_array = preprocess_image(url)
    prediction = predict(img_array)
    return prediction