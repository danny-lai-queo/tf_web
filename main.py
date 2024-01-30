from flask import Flask, render_template, request
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import cv2
import numpy as np

from flask_cors import CORS, cross_origin

import tensorflow as tf

from datetime import datetime

import base64
import os

#names = ["daisy", "dandelon", "roses", "sunflowers", "tulips"]
class_names = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']



# Process image and predict label
def processImg(img_path):
    img_height = 180
    img_width = 180
    img = tf.keras.utils.load_img(img_path, target_size=(img_height, img_width))

    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    TF_MODEL_FILE_PATH = 'model.tflite' # The default path to the saved TensorFlow Lite model
    interpreter = tf.lite.Interpreter(model_path=TF_MODEL_FILE_PATH)
    sig_dict = interpreter.get_signature_list()
    print(f"sig_dict = {sig_dict}")
    sig = list(sig_dict)[0]
    print(f'sig = {sig}')
    classify_lite = interpreter.get_signature_runner('serving_default')
    print(classify_lite)
    predictions_lite = classify_lite(sequential_1_input=img_array)['outputs']
    score_lite = tf.nn.softmax(predictions_lite)
    label_name = class_names[np.argmax(score_lite)]
    confidence_percent = 100 * np.max(score_lite)
    print("This image most likely belongs to {} with a {:.2f} percent confidence.".format(label_name, confidence_percent))

    return label_name, confidence_percent

def processImg_old(IMG_PATH):
    # Read image
    model = load_model("flower.model")
    
    # Preprocess image
    image = cv2.imread(IMG_PATH)
    image = cv2.resize(image, (199, 199))
    image = image.astype("float") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)

    res = model.predict(image)
    label = np.argmax(res)
    print("Label", label)
    labelName = class_names[label]
    print("Label name:", labelName)
    return labelName


# Initializing flask application
app = Flask(__name__)
cors = CORS(app)

@app.route("/")
def main():
    # return """
    #     Application is working
    # """
    return render_template("index.html")

# About page with render template
@app.route("/about")
def postsPage():
    return render_template("about.html")

# Process images
@app.route("/process", methods=["POST"])
def processReq():
    data = request.files["fileToUpload"]
    new_filepath = datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')
    # data.save("img.jpg")
    data.save(new_filepath)

    with open(new_filepath, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')

    #flower_name, confidence_percent = processImg("img.jpg")
    flower_name, confidence_percent = processImg(new_filepath)

    confidence_percent_str = "{:.2f}".format(confidence_percent)

    os.remove(new_filepath)

    #return flower_name
    return render_template("response.html", flower_name=flower_name, confidence_percent_str=confidence_percent_str, img_base64_str=encoded_string)


if __name__ == "__main__":
    app.run(debug=True)
