<h1>Dog Breed Prediction App</h1>
<p> Documentation prepared by <a href ="https://github.com/PhilipPeprahOwusu">Philip Peprah Owusu</a></p>
The Dog Breed Prediction project utilizes machine learning techniques to accurately classify different dog breeds based on input images. The project involves training a model using a dataset of dog images and their corresponding breeds. The code provided offers a step-by-step guide on how to access the dataset, preprocess the data, build a convolutional neural network model, and train it on the dataset. The model's performance is evaluated, and the accuracy of breed predictions on a test set is displayed. The project's goal is to showcase the process of developing a machine learning model for dog breed classification and encourage further exploration in this domain.

below is a step by step detail of what I did

<h1>Getting Started</h1>
  <p>To get started with this project, these are the steps I used:</p>
  <ol>
    <li>Uploaded the Kaggle JSON file to access the dataset.</li>
    <li>Created a directory to store the Kaggle JSON file using the command<code>!pip install -q kaggle</code>.</li>
    <li>Created a directory to store the Kaggle JSON file using the command <code>!mkdir -p /root/.kaggle</code>.</li>
    <li>Moved the Kaggle JSON file to the created directory using the command <code>!mv kaggle.json /root/.kaggle/</code>.</li>
    <li>Lastly, I set the appropriate permissions for the Kaggle JSON file using the command <code>!chmod 600 /root/.kaggle/kaggle.json</code>.</li>
  </ol>
 
 <h1>Dataset Preparation</h1>
  <ol>
    <li>Created a directory to store the dataset using the command <code>!mkdir dog_dataset</code>.</li>
    <li>Found the required dataset from Kaggle using the command <code>!kaggle datasets list -s dogbreedidfromcomp</code>.</li>
    <li>Downloaded the dataset using the command <code>!kaggle datasets download catherinehorng/dogbreedidfromcomp</code>.</li>
    <li>Unzipped the downloaded data and remove unusable files using the commands <code>!unzip dog_dataset/dogbreedidfromcomp.zip -d dog_dataset</code> and <code>!rm dog_dataset/dogbreedidfromcomp.zip</code> respectively.</li>
  </ol>
  
  
  <h1>Data Preprocessing</h1>
  <p>To preprocess the data, I performed the following </p>
  <ol>
    <li>Imported the required libraries, including numpy, pandas, matplotlib.pyplot, tqdm, keras.utils, label_binarize, train_test_split, Sequential, Dense, Dropout, Flatten, Conv2D, MaxPool2D, Adam, and load_img from tensorflow.keras.preprocessing.image.</li>
    <li>Read the labels from the CSV file using <code>dog_labels = pd.read_csv("dog_dataset/labels.csv")</code>.</li>
    <li>Counted the number of occurrences for each breed in the dataset using <code>breed_count = breed.value_counts()</code>.</li>
    <li>Selected the desired breeds for classification and create a new DataFrame using <code>labels = dog_labels[(dog_labels['breed'].isin(dog_names))]</code>.</li>
    <li>Created a numpy matrix filled with zeros to store the image data using <code>X_data = np.zeros((len(labels), 224, 224, 3), dtype='float32')</code>.</li>
    <li>Performed one-hot encoding on the labels using <code>Y_data = label_binarize(labels['breed'], classes=dog_names)</code>.</li>
    <li>Read and convert each image to a numpy array, normalize the dataset, and store it in the previously created numpy matrix using a for loop.</li>
    <li>Printed the shape and size of the train images and one-hot encoded output.</li>
  </ol>
  
  <h1>Model Building</h1>
  <p>To build the model, this is what I did</p>
  <ol>
    <li>Built a Sequential model using keras.layers, including Conv2D, MaxPool2D, Flatten, and Dense layers.</li>
    <li>Compiled the model using categorical_crossentropy as the loss function, Adam optimizer, and accuracy as the metric.</li>
    <li>Split the dataset into training and testing datasets using train_test_split.</li>
    <li>Further split the training dataset into training and validation datasets.</li>
    <li>Trained the model on the training dataset using model.fit with specified epochs and batch size.</li>
    <li>Plot the training history, showing the model accuracy for both training and validation datasets.</li>
  </ol>
  <h1>Model Evaluation</h1>
  <p>To evaluate the trained model, follow these steps:</p>
  <ol>
    <li>Evaluate the trained model on the test dataset using model.evaluate.</li>
    <li>Plot an image from the test dataset to visually compare.</li>
    <li>Print the original and predicted breed for the selected image.</li>
  </ol>

  <h1>Saving the Model</h1>
  <p>The trained model can be saved using the command model.save("dog_breed.h5").</p>

  <p>Please note that this README file serves as an overview of the code and provides step-by-step details on what I did on the project</p>
