from flask import Flask, request, redirect, url_for, render_template
from keras.models import load_model 
import numpy as np 
from keras.preprocessing import image
import cv2
import boto3
from werkzeug.utils import secure_filename
application = Flask(__name__)
global model 
model = load_model('my_model.h5') 
@application.route('/')
def index():
    return render_template('index.html')

@application.route('/predict',methods=['POST'])
def predict_vechile():
    img_file=request.files.get('file')
    filename = secure_filename(img_file.filename)
    img_file.save(filename)
    npimg = np.fromfile(img_file, np.uint8)
    #img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    access_key='AKIAVGAQOTNHLJXI5GFZ'
    secret_access_key='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'
    client=boto3.client('s3',aws_access_key_id=access_key,aws_secret_access_key=secret_access_key)
    upload_file_bucket='img-cloud'
    upload_file_key='images/'+ str(filename)
    #client.create_bucket(Bucket=upload_file_bucket)
    client.upload_file(filename,upload_file_bucket,upload_file_key)
    img= image.load_img(filename, target_size=(224, 224))
    img=image.img_to_array(img)
    img=np.expand_dims(img,axis=0)
    probabilities = model.predict(img)
    print(probabilities)
    number_to_class = ['Car','Plane']
    index = np.argmax(probabilities,axis=1)
    predictions = {
        "class":number_to_class[index[0]],
      }
    return render_template('predict.html', predictions=predictions)

if __name__=='__main__':
  application.run(host='0.0.0.0',port=8080)