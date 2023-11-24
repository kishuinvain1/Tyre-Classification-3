import io
import streamlit as st
from roboflow import Roboflow
from pathlib import Path
import os
from PIL import Image
import cv2
import numpy as np
import base64
import logging
import subprocess
from urllib.parse import quote
import json


def curl_command_hosted(url):
    print("<<<<<<<Inside curl_command>>>>>>>>>>>")
    # Your image URL
    image_url = url

    # Encode the image URL
    encoded_url = quote(image_url)

    # Construct the cURL command with the encoded URL as a variable
    bash_command = f'curl -X POST "https://detect.roboflow.com/detection-tyre/1?api_key=0Uglhm9vMkjvOzEnA7t2&image={encoded_url}"'

    process = subprocess.Popen(bash_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, error = process.communicate()

    if output:
        print("Output:", output.decode())
        #st.write("Output:", output.decode())
        return json.loads(output.decode())
    if error:
        print("Error:", error.decode())
        st.write("Error:", error.decode())
        return error.decode()


def curl_command():
    print("<<<<<<<Inside curl_command>>>>>>>>>>>")
    
    # Replace 'YOUR_BASH_COMMAND_HERE' with your actual bash command
    bash_command = 'base64 "main_image.jpg" | curl -d @- "https://detect.roboflow.com/detection-tyre/1?api_key=0Uglhm9vMkjvOzEnA7t2"'

    # Execute the bash command
    process = subprocess.Popen(bash_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, error = process.communicate()

    if output:
        print("Output:", output.decode())
        #st.write("Output:", output.decode())
        return json.loads(output.decode())
    if error:
        print("Error:", error.decode())
        st.write("Error:", error.decode())
        return error.decode()


def load_image():
    opencv_image_resz = None
    opencv_image = None 
    path = None
    f = None
    uploaded_file = st.file_uploader(label='Pick an image to test')
    print(uploaded_file)
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)
        image_data = uploaded_file.getvalue() 
        #st.image(image_data)
        name = uploaded_file.name
        path = os.path.abspath(name)
        print("abs path")
        print(path)
        opencv_image_resz = cv2.resize(opencv_image.copy(), (640,640))
        cv2.imwrite("main_image_original.jpg", opencv_image)
        cv2.imwrite("main_image.jpg", opencv_image_resz)
        #st.image("main_image.jpg", caption="svd_image")
       
    return path, opencv_image_resz
       


	


# convert PIL Image to an RGB image( technically a numpy array ) that's compatible with opencv
def toRGB(image):
    return np.array(image)
	

	
def drawBoundingBox(saved_image ,x, y, w, h, cl, cf):
    #img = Image.open(saved_image)
    #img = cv2.imread(saved_image)
    img = saved_image
    
    x = int(x)
    y = int(y)
    w = int(w)
    h = int(h)
    start_pnt = (x-w//2,y-h//2)
    end_pnt = (x+w//2, y+h//2)
    txt_start_pnt = (x-w//2, y-h//2-5)
    
    img = cv2.rectangle(img, start_pnt, end_pnt, (0,255,0), 3)
    img = cv2.putText(img, cl, txt_start_pnt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)	
    	
    return img
    


def predict(model, url):
    return model.predict(url, confidence=40, overlap=30).json()
    #return model.predict(url, hosted=True).json()
	
	
def main():
    st.title('Tyre Classification')
    option = st.selectbox('Choose Model', ('ktm-rc-200', 'ktm-duke-250'))

    image, svd_img = load_image()

    result = st.button('Detect')
    seal_bear = False
    nok_flag = False
    if result:
        #results = predict(model, svd_img)
        #results = predict(model2, url)
        results = curl_command()
        print("Prediction Results are...")	
        print(results)
        if len(results['predictions']) == 0:
            st.image(svd_img)
            st.write("No Tyre Detected")
        else:
            st.write('DETECTION RESULTS')
           
            svd_img = cv2.cvtColor(svd_img,cv2.COLOR_BGR2RGB)
            for cnt,item in enumerate(results['predictions']):
                #new_img_pth = results['predictions'][0]['image_path']
                x = results['predictions'][cnt]['x']
                y = results['predictions'][cnt]['y']
                w = results['predictions'][cnt]['width']
                h = results['predictions'][cnt]['height']
                cl = results['predictions'][cnt]['class']
                cnf = results['predictions'][cnt]['confidence']
                if 'bearing' in cl:
                    cl = cl + " & seal"
                    seal_bear = True
                elif 'seal' in cl:
                    continue    
                svd_img = drawBoundingBox(svd_img,x, y, w, h, cl, cnf)
                if not (option in cl):
                    nok_flag = True
                    break

            if nok_flag:
                st.write("*Wheel's Model is not ", option ) 
                return   

            st.image(svd_img, caption='Resulting Image') 

            #Tyre-Type
            if "ktm-rc-200-rw" in cl:
                st.write('*KTM RC 200 REAR-WHEEL')   
            elif "ktm-rc-200-fw" in cl:
                st.write('*KTM RC 200 FRONT-WHEEL') 
            elif "ktm-duke-250-rw" in cl:
                st.write('*KTM DUKE 250 REAR-WHEEL')
            elif "ktm-duke-250-fw" in cl:
                st.write('*KTM DUKE 250 FRONT-WHEEL') 


            #Company Name
            st.write('*Mfr: MRF')

            #Size
            if "ktm-rc-200-rw" in cl:
                st.write('*Size: 150/60 R17')   
            elif "ktm-rc-200-fw" in cl:
                st.write('*Size: 110/70 R17') 
            elif "ktm-duke-250-rw" in cl:
                st.write('*Size: 150/60 ZR17')
            elif "ktm-duke-250-fw" in cl:
                st.write('*Size: 110/70 ZR17') 


            #Rim Specs
            if "ktm-rc-200-rw" in cl:
                st.write('*Rim: Black Colored')   
            elif "ktm-rc-200-fw" in cl:
                st.write('*Rim: Black Colored') 
            elif "ktm-duke-250-rw" in cl:
                st.write('*Rim: Saffron Colored')
            elif "ktm-duke-250-fw" in cl:
                st.write('*Rim: Saffron Colored')  


            #Disc Brake Specs
            if "ktm-rc-200-rw" in cl:
                st.write('*Disc Brake: 5 Bolts')   
            elif "ktm-rc-200-fw" in cl:
                st.write('*Disc Brake: 5 Bolts') 
            elif "ktm-duke-250-rw" in cl:
                st.write('*Disc Brake: 5 Bolts')
            elif "ktm-duke-250-fw" in cl:
                st.write('*Disc Brake: 5 Bolts')   


            #Sensor Disc Specs
            if "ktm-rc-200-rw" in cl:
                st.write('*Sensor Disc: 5 Bolts, Black Colored')   
            elif "ktm-rc-200-fw" in cl:
                st.write('*Sensor Disc: 5 Bolts, Black Colored') 
            elif "ktm-duke-250-rw" in cl:
                st.write('*Sensor Disc: 5 Bolts, Black Colored')
            elif "ktm-duke-250-fw" in cl:
                st.write('*Sensor Disc: 5 Bolts, Black Colored')   


            #Seal & Bearing Sepcs
            if "ktm-rc-200-rw" in cl and seal_bear:
                st.write('*Seal & Bearing: Present')   
            elif "ktm-rc-200-fw" in cl and seal_bear:
                st.write('*Seal & Bearing: Present') 
            elif "ktm-duke-250-rw" in cl and seal_bear:
                st.write('*Seal & Bearing: Present')
            elif "ktm-duke-250-fw" in cl and seal_bear:
                st.write('*Seal & Bearing: Present')           

            
                     


           

if __name__ == '__main__':
    main()
