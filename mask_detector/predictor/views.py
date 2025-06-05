from django.shortcuts import render

# Create your views here.
import base64
from io import BytesIO
from PIL import Image
import numpy as np
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from tensorflow import keras

model = keras.models.load_model("face_mask_detection_model.keras")
class_names = ['Without Mask', 'With Mask']  # Adjust if class order was incorrect
THRESHOLD = 0.6  # Confidence threshold

def preprocess_image(img):
    img = img.resize((128, 128), Image.BILINEAR).convert('RGB')
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def index(request):
    result = None
    confidence = None
    image_url = None
    prediction_details = None

    if request.method == 'POST':
        if 'webcam_image' in request.POST:
            data_url = request.POST['webcam_image']
            header, encoded = data_url.split(",", 1)
            binary_data = base64.b64decode(encoded)
            img = Image.open(BytesIO(binary_data))
            img_array = preprocess_image(img)
            prediction = model.predict(img_array)[0]
            confidence_scores = {class_names[i]: float(prediction[i]) * 100 for i in range(len(class_names))}
            class_index = np.argmax(prediction)
            confidence_score = prediction[class_index]
            confidence = confidence_score * 100
            result = class_names[class_index] if confidence_score >= THRESHOLD else "Uncertain"
            prediction_details = confidence_scores

        elif request.FILES.get('image'):
            image = request.FILES['image']
            fs = FileSystemStorage()
            filename = fs.save(image.name, image)
            img_path = fs.path(filename)
            img = Image.open(img_path)
            img_array = preprocess_image(img)
            prediction = model.predict(img_array)[0]
            confidence_scores = {class_names[i]: float(prediction[i]) * 100 for i in range(len(class_names))}
            class_index = np.argmax(prediction)
            confidence_score = prediction[class_index]
            confidence = confidence_score * 100
            result = class_names[class_index] if confidence_score >= THRESHOLD else "Uncertain"
            image_url = fs.url(filename)
            prediction_details = confidence_scores

    return render(request, 'index.html', {
        'result': result,
        'confidence': confidence,
        'threshold': THRESHOLD * 100,
        'image_url': image_url,
        'prediction_details': prediction_details
    })
