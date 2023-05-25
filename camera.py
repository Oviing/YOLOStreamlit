import streamlit as st
from PIL import Image
import numpy as np
import base64

import streamlit as st
import base64
from PIL import Image
import io

import streamlit as st
import base64
from PIL import Image
import io
from ultralytics import YOLO
import requests
import json
import gdown
url = "https://yolo-syntax-dev.cfapps.eu20.hana.ondemand.com/predict"

drive = "https://drive.google.com/file/d/1oMa10Hcx_Ikrf8s85Qyut2g1ASxYgDEg/view?usp=share_link"
output = "yoloLEGO.pt"
m = gdown.download(drive, output, quiet=False)

def pil_to_numpy(image):
    """
    Converts a PIL Image to a NumPy array.

    Args:
        image (PIL.Image.Image): The input PIL Image.

    Returns:
        numpy.ndarray: The converted NumPy array.
    """
    return np.array(image)

def get_base64_from_image(image):
    buffered = image.convert("RGB")
    buffered.save("temp_image.jpg", format="JPEG")
    with open("temp_image.jpg", "rb") as file:
        base64_string = base64.b64encode(file.read()).decode("utf-8")
    return base64_string

def convert_bytesio_to_image(bytesio):
    if bytesio is not None:
        bytesio.seek(0)
        image = Image.open(io.BytesIO(bytesio.read()))
        return image
    return None

def numpy_array_to_image(array):
    # Create PIL image from the NumPy array
    image = Image.fromarray(array)

    return image

def numpy_array_to_base64(array):
    # Convert NumPy array to bytes
    array_bytes = array.tobytes()

    # Encode the bytes as Base64
    base64_str = base64.b64encode(array_bytes).decode('utf-8')

    return base64_str

def predictNCCodes(y_hat):
    classes = ["missingWheel", "wheel"]
    ncCode = ["MISSINGWHEEL"]

    l = []
    n = y_hat.cls.tolist()
    print("Number of results:", len(n))
    print(y_hat)
    for i in range(0, len(n)):
        score = float(y_hat.conf[i])
        predClass = int(y_hat.cls[i])
        coord = y_hat.xywhn[i].tolist()
        print("Score", score)
        print("Class", predClass)

        if predClass == 0:
            d = {
            "isLogged": True,
            "ncCode": ncCode[predClass],
            "predictionBoundingBoxCoords": "TEST",
            "predictionClass": classes[predClass],
            "predictionScore": score
            }
            l.append(d)
    print("result", l)
    return l

def calculateResult(result, index):

    score = float(result.conf[index])
    predictedClass = int(result.cls[index])
    coordinates = result.xywhn[index].tolist()

    return score, predictedClass, coordinates


st.title("Camera Input App")
st.write("Click the 'Capture' button to take a photo with your camera.")
nccode = st.checkbox('Auto Log NC Code')
sfc = st.text_input(label= "SFC")
bytesio = st.camera_input("Take a picture")


if nccode:
    logNCCode = True
    if bytesio:
        st.image(bytesio)
        img = convert_bytesio_to_image(bytesio)
        model = YOLO(m)
        image = pil_to_numpy(img)
        results = model(image)

        #res_plotted = results[0].plot()
        predictedBase64 = numpy_array_to_base64(results[0].plot())
        y_hat = results[0].boxes

        coord = predictNCCodes(y_hat)

        
        st.image(results[0].plot())

        returnJSON = {
                        "image" : predictedBase64,
                        "logNC" : logNCCode,
                        "coordinates" : coord,
                        "sfc" : sfc
                    }
        payload = json.dumps(returnJSON)
        response = requests.post(url, data=payload)

# Check the response status code
        if response.status_code == 200:
            # Request was successful
            print("POST request was successful.")
        else:
            # There was an error with the request
            print("POST request failed.")

        # Print the response content
        print(response.text)
        

else:
    logNCCode = False
    if bytesio:
        st.image(bytesio)
        img = convert_bytesio_to_image(bytesio)
        resized_img = img.resize((320, 320))
        model = YOLO('/home/joel/Syntax/VI/yoloLEGO.pt')
        image = pil_to_numpy(resized_img)
        
        results = model(image)

        res_plotted = results[0].plot()
        y_hat_image = Image.fromarray(res_plotted)

        with io.BytesIO() as buffer:
            y_hat_image.save(buffer, format = "PNG")
            image_bytes = buffer.getvalue()
            base64str = base64.b64encode(image_bytes).decode('utf-8')
        predictedBase64 = base64str
        

        
        st.image(results[0].plot())


        returnJSON = {
                        "image" : predictedBase64,
                        "logNC" : logNCCode,
                        "sfc" : sfc
                    }
        payload = json.dumps(returnJSON)
        headers = {
            'Content-Type': 'application/json'
        }
        response = requests.post(url, headers = headers, data=payload)

# Check the response status code
        if response.status_code == 200:
            # Request was successful
            print("POST request was successful.")
        else:
            # There was an error with the request
            print("POST request failed.")

        # Print the response content
        print(response.text)
        st.json(returnJSON)
        
