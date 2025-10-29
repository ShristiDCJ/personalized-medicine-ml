# personalized-medicine-ml
Deep Learning Project

The "Deep_Learning_Project.ipynb" file is the training model for the Personalized Medical Project.

The dataset used here is available on the given link:- https://www.kaggle.com/c/msk-redefining-cancer-treatment

The files form the dataset currently used are:-
1. training_variants.csv
2. training_text.csv

How to run the code:-
1. Download Dataset from the Link.
2. Unzip the dataset and find the above mentioned files.
3. Check if the files have correct extension, if not rename the file to have ".csv" extension.
4. If running on local environments like VS code or Curser then make sure the files are in the same folder. Copy the file path and paste in the designated line:- "variants = pd.read_csv('/content/training_variants.csv')" & "text = safe_load_text('/content/training_text.csv')".
5. If you are running the code in colab, then upload the file in the runtime.
6. Make sure you have all the necessary libraries download.
7. Run the code.
