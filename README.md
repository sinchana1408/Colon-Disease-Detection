**Colon Disease Detection using CNN**
A simple deep learning project to classify colonoscopy images into one of four medical conditions using a Convolutional Neural Network (CNN).
**📌 Overview**
This project helps identify colon-related diseases from medical images using a CNN model. 
**The model is trained on a dataset containing colonoscopy images categorized into:**
- Normal
- Ulcerative Colitis
- Polyps
- Esophagitis
**🧬 Dataset Structure**
The dataset is organized into four folders:
- 0_normal
- 1_ulcerative_colitis
- 2_polyps
- 3_esophagitis
Each folder contains images of size 512x512 pixels representing the respective disease class.
**🧠 How It Works**
- A CNN model is trained using TensorFlow/Keras to classify input colon images into one of the four classes.
- Once trained, the model can be used to predict the disease from a new colonoscopy image.
**🗂️ Folder Structure**
![Folder](https://github.com/sinchana1408/Colon-Disease-Detection/blob/6c7e59284b44f416e04eefea3ae7744b8a1f7b37/Screenshot%202025-07-12%20202951.png)
**🚀 How to Run**
Step 1: Clone the Repository
git clone https://github.com/sinchana1408/Colon-Disease-Detection.git
cd Colon-Disease-Detection
Step 2: Install Dependencies
pip install -r requirements.txt
Step 3: Train the Model (optional)
python train.py
> This will create a colon_model.h5 file after training.
Step 4: Run Prediction
python app.py
You will be prompted to select an image, and the model will output the predicted class.
**✅ Example Prediction**
Input Image: sample_image.jpg
Output: Predicted class: Ulcerative Colitis
> **Screenshots**
> ![Alt Text](https://github.com/sinchana1408/Colon-Disease-Detection/blob/6c7e59284b44f416e04eefea3ae7744b8a1f7b37/Screenshot%202025-07-12%20133718.png)
>
>![Home Page](https://github.com/sinchana1408/Colon-Disease-Detection/blob/6c7e59284b44f416e04eefea3ae7744b8a1f7b37/Screenshot%202025-07-12%20133847.png)
**🧪 Technologies Used**
- Python
- TensorFlow / Keras
- NumPy
- OpenCV / PIL (for image processing)
📃 License
This project is licensed under the MIT License.
👤 Author
Sinchana Shivanand
GitHub: https://github.com/sinchana1408

