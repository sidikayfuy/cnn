import io

from django.shortcuts import render
import base64
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from keras.models import load_model
import numpy as np
from PIL import Image

try:
    complete_model = load_model("/content/cnn/complete_model.h5")
except Exception:
    complete_model = load_model("./complete_model.h5")

def index(request):
    return render(request, 'paint/index.html', {})


@csrf_exempt
def sendpaint(request):
    try:
        image_data = request.POST['image_data']
        image_data = image_data.split(',')[1]
        image_binary = base64.b64decode(image_data)
        img = Image.open(io.BytesIO(image_binary))
        img = img.convert("RGBA")
        np_img = np.array(img)
        black_pixels = (np_img[..., :3] == [0, 0, 0]).all(axis=2)
        np_img[black_pixels, :3] = [255, 255, 255]
        black_bg = Image.new("RGBA", img.size, (0, 0, 0, 255))
        image = Image.alpha_composite(black_bg, Image.fromarray(np_img))
        image = image.resize((28, 28)).convert('L')
        image_array = np.array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        prediction = complete_model.predict(image_array)
        predicted_digit = np.argmax(prediction)
        return JsonResponse({'status': 'success', 'predicted_digit': int(predicted_digit)})

    except Exception as e:
        return JsonResponse({'status': 'error', 'message': str(e)})

