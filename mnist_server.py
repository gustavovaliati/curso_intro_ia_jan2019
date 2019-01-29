import keras
from keras.models import model_from_json

import base64
import numpy as np
from PIL import Image
from io import BytesIO

from flask import Flask, request, send_from_directory

app = Flask(__name__)

json_file = open('mnist_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
model.load_weights("mnist_model.h5")

@app.route('/', methods=['GET'])
def index():
    return send_from_directory('.', 'mnist_web_app.html')


@app.route('/mnist', methods=['POST'])
def mnist():
    data = request.form['img']
    encoded_image = data.split(",")[1]
    im = Image.open(BytesIO(base64.b64decode(encoded_image)))
    # im.show()
    im = im.resize((28,28), Image.ANTIALIAS)
    img = np.array(im, np.float32) / 255
    print(img)
    print(img.shape)

    test_img = np.expand_dims([img[:,:,3]], axis=3)
    print(test_img.shape)
    print(test_img)

    pred_y = model.predict(test_img)
    print(pred_y)

    return 'I guess you have drawn a: ' + str(np.argmax(pred_y[0]))
