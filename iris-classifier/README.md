# Iris Classifier
This is a simple Iris flower classifier for the CPE393 ML deployment assignment. This use a simple Random Forest Classifier to classify the Iris flower species based on the features of the flower.

## Training
To train the model, run the following command:
```bash
python ./model/classifer.py
```
This will train the model and save it to the `app` directory.

## Development
To run the app, run the following command:
```bash
fastapi dev ./app/main.py
```
This will start the FastAPI server on `http://localhost:8000`.

## Deployment
To deploy the app, run the following command:
```bash
docker build -t iris-classifier .
docker run -p 8000:80 iris-classifier
```
This will build the Docker image and run the container on port 8000.
