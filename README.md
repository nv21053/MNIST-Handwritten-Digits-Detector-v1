# MNIST Handwritten Digits Detector

This project is a web application that uses a pre-trained machine learning model to detect handwritten digits from images. It is implemented using Flask, TensorFlow, and Bootstrap.

## Setup

1. Clone the repository:

```bash
git clone https://github.com/nv21053/MNIST-Handwritten-Digits-Detector.git
```

2. Install the dependencies:

```bash
pip install -r requirements.txt
```

3. Download the pre-trained model file (`mnist_model.h5`) and place it in the project root directory.

## Usage

1. Run the Flask web server:

```bash
python app.py
```

2. Open your web browser and navigate to `http://localhost:5000`.

3. Choose an image of a handwritten digit and click the "Submit" button.

4. The application will predict the digit and display the result along with the probabilities for each digit.

5. To try again, click the "Try Again" button.

## File Structure

- `app.py`: The Flask application script.
- `mnist_model.h5`: The pre-trained machine learning model for digit recognition.
- `templates/index.html`: The HTML template for the web application.

## Dependencies

- Flask: Web framework for Python.
- TensorFlow: Deep learning library for machine learning tasks.
- Pillow: Python Imaging Library for image processing.
- Flask-Bootstrap: Integration of Bootstrap with Flask.

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvement, please create an issue or submit a pull request.



# Python Flask Example

This is a [Flask](https://flask.palletsprojects.com/en/1.1.x/) app that serves a simple JSON response.

[![Deploy on Railway](https://railway.app/button.svg)](https://railway.app/new/template/zUcpux)

## ✨ Features

- Python
- Flask

## 💁‍♀️ How to use

- Install Python requirements `pip install -r requirements.txt`
- Start the server for development `python3 main.py`
