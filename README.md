# Digit Recogniser
This is a digit recogniser web app. It consists of a model backend, 
web app and database services. Docker Compose defines the
multi-container set up.

## Model Backend
The model backend contains FastAPI service which serves a PyTroch digit recogniser 
model.

The package also contains the model architecture and the code used for training. 
The model is a mini-ResNet, trained for 3 epochs on the MNIST dataset,
achieving over 99% accuracy on the test data.

## Web App
A simple Streamlit web app providing a user interface for the digit recogniser. 

## Database
A PostgresSQL database where the app usage history is stored.

## Deployment
The app is deployed on a GCP instance, and can be accessed at: http://35.197.194.75:8501/ 
