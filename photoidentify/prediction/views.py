import base64
from django.shortcuts import render
from .forms import ImageUploadForm
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
import numpy as np
from io import BytesIO

def predict(request):
    if request.method == 'GET':
        form = ImageUploadForm()
        return render(request, 'home.html', {'form': form})

    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            img_file = form.cleaned_data['image']
            img_file = BytesIO(img_file.read())
            img = load_img(img_file, target_size=(224, 224))
            img_array = img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)

            model_path = '/Users/kamakuraryouma/Desktop/butikou/photoidentify/prediction/models/vgg16.h5'
            model = load_model(model_path)
            predictions = model.predict(img_array)
            decoded_predictions = decode_predictions(predictions, top=5)[0]

            prediction_data = [{'label': pred[1], 'probability': pred[2]} for pred in decoded_predictions]

            img_file.seek(0)
            img_data = base64.b64encode(img_file.read()).decode('utf-8')

            return render(request, 'home.html', {'form': form, 'prediction_data': prediction_data, 'img_data': img_data})
        else:
            return render(request, 'home.html', {'form': form})


