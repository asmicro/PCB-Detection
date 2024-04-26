import os
import subprocess
import streamlit as st
import sys
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
from image_label import image_label_component,load_image,img_to_base64
from table import create_df

st.set_page_config(page_title="PCB detection", layout="wide",page_icon="assets/icon.png")


def format_change(path):
    import random

    out=[]
    with open (path) as f:
        result = [line for line in f.readlines() if line.strip()]

    for a in result:  
        lis=list(map(float,a.split(" ")))
        
        names= ["Missing Holes",
        "Mouse Bites",
        "Open Circuit",
        "Short",
        "Spur",
        "Spurious Copper"]
        
        dict1 = {"type": "RECTANGLE"}
        dict1["x"]=(lis[1]-(lis[3]/2))*100
        dict1["y"]=(lis[2]-(lis[4]/2))*100
        dict1['width']=lis[3]*100
        dict1['height']=lis[4]*100
        dict2={"text": names[int(lis[0])]}
        # path="../data"
        dict2["id"]=random.random()
        out_dict={"geometry":dict1, "data":dict2}
        
        out.append(out_dict)
        # print(out)
    return(out)

def run_image_processing(file):
    # Define your image processing logic here
    cache = 'cache'
    weights = "pcb_detection.pt" 
    detectorScript = "detect.py"

    cacheAbsolutePath = os.path.join(os.getcwd(), cache)
    if not os.path.exists(cacheAbsolutePath):
        os.makedirs(cacheAbsolutePath)
    detectPath = os.path.join(cacheAbsolutePath, 'detect')
    if not os.path.exists(detectPath):
        os.makedirs(detectPath)
    filepath = os.path.join(cacheAbsolutePath, file)
    try:
        python_executable = sys.executable
        #add iamge size
        subprocess.run([python_executable , detectorScript, '--source', filepath, '--weights', weights, '--conf', '0.25', '--name', 'detect', '--exist-ok', '--project', cacheAbsolutePath, "--no-trace" ,"--save-txt", ])
    
        # subprocess.run([python_executable, detectorScript, "--weights", weights, "--source", filepath, "--img-size", "416", "--save-txt", "--save-conf", "--save-crop", "--nosave", "--exist-ok", "--project", cacheAbsolutePath, "--name", "detect"])
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running the subprocess: {e}")
        return False
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return False
    return True

def resize_image(image_path, max_width):

    # Open the image using Pillow
    image = Image.open(image_path)

    # Get the original dimensions
    width, height = image.size

    # Calculate the new height while maintaining the aspect ratio
    new_width = max_width
    new_height = int(height * (new_width / width))

    # Resize the image
    resized_image = image.resize((new_width, new_height), Image.ANTIALIAS)

    # Save the resized image
    resized_image.save(image_path)



def process_uploaded_image(uploaded_file):
    
    saveDir = os.path.join(os.getcwd(), 'cache')

    if not os.path.exists(saveDir):
        os.makedirs(saveDir)
    

    fileExtension=uploaded_file.name.split(".")[-1]
    fileName = "1."+fileExtension

    savePath = os.path.join(saveDir, fileName)

    with open(savePath, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # resize_image(savePath, 1000)
    
    # //removing last detection label folder
    
    labelPath = os.path.join(os.getcwd(),"cache/detect/labels/1.txt")
    
    if os.path.exists(labelPath):
        os.remove(labelPath)

    reponse = run_image_processing(fileName)

    if not reponse:
        return False

    output_filepath=os.path.join(os.getcwd(),"cache/detect/labels",fileName.replace(fileExtension,"txt"))
    responseFromModel,preview = st.tabs(["Response From Model","Preview"])
    with responseFromModel:
        st.markdown("### Successfully processed the image")
        st.success("Go to preview tab to see the output")
    with preview:
        handle_last_detection_edit(uploaded_file,key="last-detection")
    return True


def handle_detection_ui():
    col1,col2= st.columns(2)
    with col1:
        uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png","webp"],key=f"fileUploader")
        # Image processing button
        response = False
        buttonPressed = st.button("Process Image",key=f"buton@fileUplaod")
    if buttonPressed and uploaded_file is not None:

        # saving this file to a temporary location
        response = process_uploaded_image(uploaded_file)
        
        # if response:
            # st.success("Image processed successfully")
        
    with col2:
        if not response and uploaded_file:
            st.caption("Uploaded Image")
            st.image(uploaded_file)
    
    return uploaded_file,response
        # if response:
        #     edit = st.button("edit")
        #     if edit:
        #         label=os.path.join(os.getcwd(),"cache/detect/labels/1.txt")
        #         detectedAnnotations = format_change(label)
        #         st.write(detectedAnnotations)
            # annotations = image_label_component(image=uploaded_file,labels=["Missing Holes","Mouse Bites" ,"Open Circuit", "Short","Spur","Spurious Copper"],detectedAnotations=detectedAnnotations,key="image-label-test")

def handle_last_detection_edit(uploaded_file,key=None):
    fileExtension=uploaded_file.name.split(".")[-1]
    fileName = "1."+fileExtension
    label=os.path.join(os.getcwd(),"cache/detect/labels/1.txt")
    imagePath = os.path.join(os.getcwd(),"cache",fileName)
    detectedAnnotations = format_change(label)
    image = load_image(imagePath)
    image_str = img_to_base64(image)
    image_label_component(key=key,image=image_str,labels=["Missing Holes","Mouse Bites" ,"Open Circuit", "Short","Spur","Spurious Copper"],detectedAnotations=detectedAnnotations)

    
def load_heading():
    st.image("assets/head.png")
    st.markdown("### ⚡️ PCB Fault Identification Application")

def load_info():
    st.markdown("Welcome to our Streamlit-based interface engineered specifically for the analysis of Printed Circuit Board (PCB) images. Our sophisticated application harnesses advanced image processing to detect an array of PCB defects with significant accuracy. ")
    st.caption("This application serves as a demonstration of the capabilities inherent in our state-of-the-art PCB defect detection model. Its primary function is to validate the model's effectiveness and showcase its potential for practical implementation.")

    st.markdown("Our system is currently equipped to identify the following categories of defects:")
    cols = st.columns(8) 
    
    defects = ["Missing Holes","Mouse Bites" ,"Open Circuit", "Short","Spur","Spurious Copper"]
    for index, defect in enumerate(defects):
        with cols[index]:
            st.info(defect)
    
    st.markdown("---")
    st.markdown("##### ⚙️ About Model")
    st.markdown("The model used for this application is a [YOLOv7](https://github.com/WongKinYiu/yolov7). The model was trained for 100 epochs. To download the weights, click [here](https://drive.google.com/file/d/1HhgrRnlixFtZlZA_E19eW7wVro818Kfk/view?usp=sharing)")
    
    st.markdown("---")
    st.markdown("##### 🛄️ About Dataset")
    st.markdown(" model trained on the [PCB dataset](https://www.kaggle.com/datasets/akhatova/pcb-defects) from kaggle.")

    st.markdown("---")
    st.markdown("##### 🌟️ Performance")
    st.caption("The model has an mAP of 0.9 on the validation set.")
    st.markdown("> Losses after 100 epochs")
    st.markdown("""
| Loss | Validation | Train |
| ----------- | ----------- | ----------- |
| Box Loss | 0.07389 | 0.01253 |
| Object Loss | 0.007394 | 0.004433 |
| Class Loss | 0.01108 | 0.0007131 |
""")
    st.markdown("> Graphs")
    cf,dumb,dumb = st.columns(3)
    with cf:
        st.image("assets/cf.png",caption="Confusion Matrics")
    pr,dumb,dumb = st.columns(3)
    with pr :
        st.image("assets/pr.png", caption="Precision Recall")

def more():

    st.markdown("### Github Repository")
    st.markdown("> [Click here]()")

    st.markdown("---")
    st.markdown("### YOLOv7")
    st.markdown("> [Click here](https://github.com/WongKinYiu/yolov7)")

    st.markdown("---")
    st.markdown("### Label Img")
    st.markdown("> A package created for labeling images in streamlit as a part of this project. [PyPi Link](https://pypi.org/project/image-label/) and [Github Link](https://github.com/SksOp/image_label)")

def main():
    tab1, tab2, tab3= st.tabs(["Introduction", "Try it out","More Details"])
    with tab1:
        
        load_heading()
        load_info()
    with tab2:
        uploadedfile,response =  handle_detection_ui()
    with tab3:
        more()

            # st.write(format_change("./cache/detect/labels/1.jpg"))
    #     annotations = image_label_component(image="assets/test.jpg",labels=["Missing Holes","Mouse Bites" ,"Open Circuit", "Short","Spur","Spurious Copper"],detectedAnotations=[],key="image-label-test")
if __name__ == "__main__":
    main()