from django.shortcuts import render
from django.http import HttpResponse
import os
import numpy as np
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
import pickle
from PIL import Image
from keras.applications.vgg19 import VGG19, preprocess_input, decode_predictions
import matplotlib.pyplot as plt
from keras.models import Model
from keras.preprocessing.image import load_img,img_to_array

from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
model = load_model('C:/Users/ASUS/Downloads/Project/Project/models/model.h5')

labels = {0: 'Healthy', 1: 'Powdery', 2: 'Rust'}

# Create your views here.

def home(request):
    return render(request, 'home.htm')

def upload_file(request):
    if request.method == 'POST':
        uploaded_file = request.FILES.get('file')  # Assuming 'file' is the name of the input field in your HTML form
        if uploaded_file:
            # Process the uploaded file (save it, perform analysis, etc.)
            # For demonstration purposes, let's just store the file name

            image_name = uploaded_file.name

            save_dir = 'C:/Users/ASUS/Downloads/Project/Project/mlapp/static/Images'  # Replace with your actual static directory
            global image_path
            image_path = os.path.join(save_dir, image_name)

            # Save the uploaded file to a directory
            with open(os.path.join(save_dir, image_name), 'wb+') as destination:
                for chunk in uploaded_file.chunks():
                    destination.write(chunk)

            context = {'image_name': image_name, 'image_path': image_path}
            predictions=getResult(image_path)
            predicted_label = labels[np.argmax(predictions)]
            # return str(predicted_label)

            # Add the rest of your caption generation logic
         

            return render(request, 'result.htm', {'predicted_label': predicted_label, 'context': context})

    return render(request, 'home.htm')

def delete_file(request):
    os.remove(image_path)
    return render(request, 'home.htm')

def getResult(image_path):
    img = load_img(image_path, target_size=(225,225))
    x = img_to_array(img)
    x = x.astype('float32') / 255.
    x = np.expand_dims(x, axis=0)
    predictions = model.predict(x)[0]
    return predictions


