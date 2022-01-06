from app import app
from flask import Flask, flash, request, redirect, render_template
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import io
from PIL import Image
import base64
from Helpers import *
from joblib import load
from skimage.filters import laplace, sobel, roberts
from skimage.feature import canny
from sklearn.preprocessing import QuantileTransformer
import matplotlib.pyplot as plt
import os

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def detect_blur_fft(image, size=60, vis=False):
  # grab the dimensions of the image and use the dimensions to
  # derive the center (x, y)-coordinates
  (h, w) = image.shape
  (cX, cY) = (int(w / 2.0), int(h / 2.0))
  # check to see if we are visualizing our output
  fft = np.fft.fft2(image)
  fftShift = np.fft.fftshift(fft)

  if vis:
    # compute the magnitude spectrum of the transform
    magnitude = 20 * np.log(np.abs(fftShift))
    # display the original input image
    (fig, ax) = plt.subplots(1, 2, )
    ax[0].imshow(image, cmap="gray")
    ax[0].set_title("Input")
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    # display the magnitude image
    ax[1].imshow(magnitude, cmap="gray")
    ax[1].set_title("Magnitude Spectrum")
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    # show our plots
    plt.show()
    # zero-out the center of the FFT shift (i.e., remove low

  # frequencies), apply the inverse shift such that the DC
  # component once again becomes the top-left, and then apply
  # the inverse FFT
  fftShift[cY - size:cY + size, cX - size:cX + size] = 0
  fftShift = np.fft.ifftshift(fftShift)
  recon = np.fft.ifft2(fftShift)
  # compute the magnitude spectrum of the reconstructed image,
  # then compute the mean of the magnitude values
  magnitude = 20 * np.log(np.abs(recon))
  # the image will be considered "blurry" if the mean value of the
  # magnitudes is less than the threshold value
  return magnitude

def get_data(image):
    lap_feat = laplace(image)
    sob_feat = sobel(image)
    rob_feat = roberts(image)
    canny_feat = canny(image)
    fft_feat = detect_blur_fft(image)
    feature = [lap_feat.var(), np.amax(lap_feat),
               sob_feat.mean(), sob_feat.var(), np.amax(sob_feat),
               rob_feat.mean(), rob_feat.var(), np.amax(rob_feat),
               canny_feat.mean(), canny_feat.var(),
               fft_feat.mean(), fft_feat.var()]
    return np.array(feature)

@app.route('/')
def upload_form():
    return render_template('mainpage.html')


@app.route('/', methods=['POST'])
def upload_image():
    images = []
    for file in request.files.getlist("file[]"):
        print("***************************")
        print("image: ", file)
        if file.filename == '':
            flash('No image selected for uploading')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filestr = file.read()
            npimg = np.frombuffer(filestr, np.uint8)
            image = cv2.imdecode(npimg, cv2.IMREAD_UNCHANGED)
            # image = Helpers.resize(image, height=500)

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            feature = get_data(gray).reshape(1, -1)

            scaler = load('scaler.joblib')
            feature = scaler.transform(feature)

            clf = load('mlp1.joblib')
            prediction = clf.predict(feature)


            result = ""

            if prediction[0] == 0:
                result = "Sharp"

            elif prediction[0] == 2:
                result = "Motion Blur"
            else:
                result = "Blur"

            message = [result]

            img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            file_object = io.BytesIO()
            img = Image.fromarray(Helpers.resize(img, width=500))
            img.save(file_object, 'PNG')
            base64img = "data:image/png;base64," + base64.b64encode(file_object.getvalue()).decode('ascii')
            images.append([message, base64img])

    print("images:", len(images))
    return render_template('mainpage.html', images=images)


if __name__ == "__main__":
    app.run(debug=True)