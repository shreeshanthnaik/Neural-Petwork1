\# 🐾 Neural Petwork (v1)



This is a web application that uses a deep learning model (a Convolutional Neural Network) to classify uploaded images as either a \*\*cat\*\* or a \*\*dog\*\*.



This project was built using Keras/TensorFlow for the model and Flask for the web interface.



\### 📈 Model Performance



This first version was built from scratch and achieved \*\*~75% accuracy\*\* on the validation dataset.



---



\### 🚀 How to Run This Project



1\.  \*\*Clone the repository:\*\*

&nbsp;   ```bash

&nbsp;   git clone \[https://github.com/YOUR-USERNAME/Neural-Petwork.git](https://github.com/YOUR-USERNAME/Neural-Petwork.git)

&nbsp;   cd Neural-Petwork

&nbsp;   ```



2\.  \*\*Install dependencies (using hatch):\*\*

&nbsp;   ```bash

&nbsp;   hatch run pip install tensorflow flask pillow matplotlib

&nbsp;   ```



3\.  \*\*Get the Data (Required for training):\*\*

&nbsp;   The image dataset is not included in this repository. You must download it from the \[Kaggle "Dogs vs. Cats" competition](https://www.kaggle.com/c/dogs-vs-cats).

&nbsp;   \* Unzip `train.zip`.

&nbsp;   \* Create the `dataset/train/cats` and `dataset/train/dogs` folders.

&nbsp;   \* Run the `prepare\_data.py` script to automatically create the validation set:

&nbsp;       ```bash

&nbsp;       hatch run python prepare\_data.py

&nbsp;       ```



4\.  \*\*Run the Web App:\*\*

&nbsp;   ```bash

&nbsp;   hatch run python app.py

&nbsp;   ```



5\.  Open `http://127.0.0.1:5000` in your web browser.

