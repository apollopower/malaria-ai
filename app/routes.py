from app import app
from neural_net.core import *

from PIL import Image
from flask import request, jsonify
from io import BytesIO

@app.before_first_request
def _load_model():
    global model

    model = conv_model('notebooks/malaria_detection.pt')

@app.route('/')
def root():
    return "Malaria API is online!"


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    classes = ['Parasite','Uninfected']

    img_url = request.args.get("image")
    
    # img_response = 'http://s3-us-west-2.amazonaws.com/seround-avatars.com/4c07f769-28d1-4e8c-9f6f-24c412133fa8'
    # image = './TEST_IMG.png'
    response = {}
    response['predict'] = predict_malaria(model, classes, img_url)
    print(response)
    return jsonify(**response)
