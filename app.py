
from flask import Flask, render_template, request, send_from_directory, url_for
import os
import cv2
import numpy as np
from keras.models import load_model
import base64
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Define the directory where uploaded images will be stored
UPLOAD_FOLDER = "static/images"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Loading the model
model = load_model("maize_leaf_disease_model.h5")

# Name of Classes
CLASS_NAMES = [
    "Common_Rust",
    "Gray_Leaf_Spot",
    "Healthy",
    "Northern_Leaf_Blight",
]

# Name of Classes and corresponding information
CLASS_INFO = {
    "Common_Rust": {
        "course": "Common Rust is a fungal disease that affects maize leaves.",
        "control_measure": "Apply fungicides and practice crop rotation.",
    },
    "Gray_Leaf_Spot": {
        "course": "Gray Leaf Spot is caused by a fungus affecting maize leaves.",
        "control_measure": "Use resistant varieties and practice proper field sanitation.",
    },
    "Healthy": {
        "course": "The maize plant is healthy with no visible signs of disease.",
        "control_measure": "Maintain good agricultural practices and monitor for pests.",
    },
    "Northern_Leaf_Blight": {
        "course": "Northern Leaf Blight is a fungal disease affecting maize plants.",
        "control_measure": "Use resistant maize varieties and practice crop rotation.",
    },
}


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        # Check if the post request has the file part
        if "image" not in request.files:
            return "No file part"
        file = request.files["image"]
        # If the user does not select a file, the browser submits an empty file without a filename
        if file.filename == "":
            return "No selected file"
        if file:
            # Save the uploaded file to the specified directory
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(file_path)

            # Read the uploaded image
            opencv_image = cv2.imread(file_path)
            opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
            opencv_image = cv2.resize(opencv_image, (256, 256))
            opencv_image = np.expand_dims(opencv_image, axis=0)

            # Make prediction
            Y_pred = model.predict(opencv_image)
            predicted_class = np.argmax(Y_pred)
            result = {
                "disease": CLASS_NAMES[predicted_class],
                "course": CLASS_INFO[CLASS_NAMES[predicted_class]]["course"],
                "control_measure": CLASS_INFO[CLASS_NAMES[predicted_class]][
                    "control_measure"
                ],
                "image": filename,
            }

            return render_template("index.html", result=result)


@app.route("/static/images/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)
