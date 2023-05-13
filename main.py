#Sources:
#https://flask.palletsprojects.com/en/2.2.x/patterns/fileuploads/
#https://thinkinfi.com/upload-and-display-image-in-flask-python/
#https://flask.palletsprojects.com/en/1.1.x/patterns/fileuploads/
#https://www.youtube.com/watch?v=GeiUTkSAJPs&ab_channel=ArpanNeupane
#https://pythonhow.com/python-tutorial/flask/Adding-CSS-styling-to-your-website/
#
import sys

sys.path.append('/usr/local/lib/python2.7/site-packages')

from flask import Flask, render_template, request, session, redirect, url_for
import numpy as np
import math
import matplotlib.pyplot as plt
from flask_wtf import FlaskForm
from werkzeug.utils import secure_filename
import os

#import cv2


import tensorflow as tf
from tensorflow import keras
import nibabel as nib
#import cv2


app = Flask(__name__)



@app.route('/', methods=["GET", "POST", "REPORT"])
def index():
  print(request.method)
  if request.method == 'POST':
    # check if the post request has the file part
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    # If the user does not select a file, the browser submits an
    # empty file without a filename.
    if file.filename == '':
        return redirect(request.url)
    if file:
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        render_template('form.html')
  elif request.method == 'REPORT':
    return redirect(url_for('home'))
  return render_template('form.html')
    

def V_prime_with_gamma(V, t, a, b, r, GAMMA):
  t_for_drug = t % 21
  if t_for_drug > 10*4: # the drug wears off after approximately 4 half lifes
    return a * (math.log(b) - math.log(V)) * V
  else:
    return a * (math.log(b) - math.log(V)) * V  - GAMMA * (math.e**(-r * (t_for_drug))) * V

def V_prime(V):
  a = .0125
  b = 25000
  return a * (math.log(b) - math.log(V)) * V

def run_gompertz(initial_vol):
  a = .0125
  b = 25000
  V = initial_vol
  
  t_values = []
  V_values = []
  
  t = 0
  dt = 0.1 #Keeep it at 1, or you have to change the ticks
  t_final = 13
  
  while t <= t_final:
    t_values.append(t)
    V_values.append(V)
    V1 = V_prime(V)
    V2 = V_prime(V + dt * V1/2)
    V3 = V_prime(V + dt * V2/2)
    V4 = V_prime(V + dt * V3)
  
  
    V += (V1/6 + V2/3 + V3/3 + V4/6) * dt
    t += dt
  



  GAMMA = 0.262
  r = math.log(2) / 1.04




  V = V_values[-1]
    
  gamma_t_values = []
  gamma_V_values = []
  
  t = 0
  dt = 0.1 #Keeep it at 1, or you have to change the ticks
  t_final = 42
  
  while t <= t_final:
    gamma_t_values.append(t)
    gamma_V_values.append(V)
    V1 = V_prime_with_gamma(V, t, a, b, r, GAMMA)
    V2 = V_prime_with_gamma(V + dt * V1/2, t, a, b, r, GAMMA)
    V3 = V_prime_with_gamma(V + dt * V2/2, t, a, b, r, GAMMA)
    V4 = V_prime_with_gamma(V + dt * V3, t, a, b, r, GAMMA)
  
  
    V += (V1/6 + V2/3 + V3/3 + V4/6) * dt
    t += dt


  V = initial_vol
    
  full_t_values = []
  full_V_values = []
  
  t = 0
  dt = 0.1 #Keeep it at 1, or you have to change the ticks
  t_final = 55
  
  while t <= t_final:
    full_t_values.append(t)
    full_V_values.append(V)
    V1 = V_prime(V)
    V2 = V_prime(V + dt * V1/2)
    V3 = V_prime(V + dt * V2/2)
    V4 = V_prime(V + dt * V3)
  
  
    V += (V1/6 + V2/3 + V3/3 + V4/6) * dt
    t += dt
    
  times = np.concatenate((np.array(t_values), np.array(gamma_t_values) + t_values[-1]))
  V_values_combined = np.concatenate((np.array(V_values), np.array(gamma_V_values)))
  
  plt.plot(full_t_values, np.array(full_V_values) / 1000, label = "Gompertz Model without Chemotherapy Component")
  
  plt.plot(times, V_values_combined / 1000, label = "Gompertz Model with Chemotherapy")
  plt.ylabel("Predicted Tumor Size (mL)")
  plt.xlabel("Time (days)")
  leg = plt.legend(loc='upper center')
  plt.ylim(ymin=0)
  plt.grid()
  plt.title("Predicted Gompertz Tumor Growth with and without Doxorubicin Component")
  plt.savefig('static/IMG/prognosis_graph.png')


IMG_FOLDER = os.path.join('static', 'IMG')

app.config['UPLOAD_FOLDER'] = IMG_FOLDER



  
  #return f"""<h2>{form_data}</h2>

#<img src="{{ img }}" alt="Tumor Prognosis">
  
#<img src="prognosis_graph.png">"""
    #return render_template("home.html")




def make_volume_pred():
  
  volume = 0
  model = keras.models.load_model("trained_segmentation_model_final_1.h5")
  
  
  flair_file = nib.load('static/IMG/flair.nii.gz')
  flair_file = np.asarray(flair_file.dataobj, dtype=np.float64)
  flair_file /= flair_file.max()
  flair_file *= 255
  
  t1_file = nib.load('static/IMG/t1.nii.gz')
  t1_file = np.asarray(t1_file.dataobj, dtype=np.float64)
  t1_file /= t1_file.max()
  t1_file *= 255
  
  t2_file = nib.load('static/IMG/t2.nii.gz')
  t2_file = np.asarray(t2_file.dataobj, dtype=np.float64)
  t2_file /= t2_file.max()
  t2_file *= 255
  
  
  
  combined_arr = np.stack((flair_file, t1_file, t2_file), axis = 3)
  print(combined_arr.shape)

  flair_file = None
  del flair_file

  t1_file = None
  del t1_file

  t2_file = None
  del t2_file
  
  
  
  for splice in range(155):
    #img_to_add = cv2.resize(combined_arr[:,:,splice,:], (128,128))
    #img_to_add = cv2.resize(combined_arr[:,:,splice,:], (128,128))
    img_to_add = combined_arr[(0+56):(128+56),(0+56):(128+56),splice,:]
    model_prediction = model.predict(np.array([img_to_add]))[0].argmax(axis=2).astype(int)
    volume += np.sum(model_prediction == 3)

    img_to_add = None
    del img_to_add
  
  
  return volume / 20.1152625 # Scaling Factor
    #Load in Nifti file user uploaded
    #Process File
    #Load in model and apply it
    #Get volume Predictions
    #Save a few segmentation colormap images for the report
    #Maybe also give volumes of other regions

@app.route("/home", methods=["GET", "POST"])
def home():
  #form_data = request.values.get('ivol')
  volume = make_volume_pred()
  run_gompertz(int(volume))

  full_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'prognosis_graph.png')
  css_loc = os.path.join(app.config['UPLOAD_FOLDER'], 'style.css')
  
  return render_template('report.html', img_file_path=full_filename, form_data=volume / 1000, css_loc=css_loc)

app.run(host='0.0.0.0', port=81)
