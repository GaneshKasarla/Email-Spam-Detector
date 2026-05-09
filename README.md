# Spam Message Detector

This is a simple machine learning project used to detect whether a message is spam or not spam (ham).

I built this project using Python, Streamlit, Pandas, and Scikit-learn.  
The model is trained using the Multinomial Naive Bayes algorithm and TF-IDF Vectorization.

## Features

- Detects spam and normal messages
- Simple and easy-to-use interface
- NLP based preprocessing
- Fast prediction system
- Machine learning based classification

## Technologies Used

- Python
- Streamlit
- Pandas
- Scikit-learn

## Project Workflow

1. Load the dataset
2. Clean and preprocess the text
3. Convert text into numerical vectors using TF-IDF
4. Train the model using Multinomial Naive Bayes
5. Predict whether the message is spam or ham

## Files in the Project

- app.py → Streamlit web application
- train_model.py → Model training file
- spam.csv → Dataset
- model.pkl → Saved trained model
- vectorizer.pkl → Saved TF-IDF vectorizer
- requirements.txt → Required libraries
- README.md → Project documentation

## How to Run the Project

Install required libraries:

pip install -r requirements.txt

Run the Streamlit app:

streamlit run app.py

## Future Improvements

- Improve UI design
- Add confidence score
- Try different ML algorithms
- Deploy the project online

## Author
Kasarla Ganesh Bharadwaj
