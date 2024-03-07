from flask import Flask, render_template, request, jsonify
import numpy as np
import tensorflow as tf
from PIL import Image
from flask_bootstrap import Bootstrap

app = Flask(__name__)
Bootstrap(app)

# Load the pre-trained model
model = tf.keras.models.load_model('mnist_model.h5')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the uploaded image file
    image = request.files['image']
    image.save('uploaded_image.png')

    # Pre-process the image
    image = Image.open('uploaded_image.png').convert('L')  # Convert to grayscale
    image = image.resize((28, 28))  # Resize to 28x28 pixels
    image = np.array(image)  # Convert to NumPy array
    image = image / 255.0  # Normalize the pixel values

    # Reshape the image for the model
    image = image.reshape(1, 28, 28, 1)

    # Make predictions
    predictions = model.predict(image)
    predicted_digit = np.argmax(predictions[0])

    # Calculate the probabilities for each digit
    probabilities = predictions[0] * 100
    probabilities = [round(p, 2) for p in probabilities]

    # Hide the submit button
    disable_submit = True

    return render_template('index.html', predicted_digit=predicted_digit, probabilities=probabilities, disable_submit=disable_submit)


if __name__ == '__main__':
    app.run(debug=True, port=os.getenv("PORT", default=5000))
