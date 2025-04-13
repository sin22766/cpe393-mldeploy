# House Price Regressor
This is a simple regressor for predicting the house price for the CPE393 ML deployment assignment. This use a  Gradient Boosting Regressor to predict the house price from the house features.

Since this app use the FastAPI instead of Flask, it can validate the input data more easily using the Pydantic model. This will return the HTTP 422 Unprocessable Entity error if the input data is not valid.

## Training
To train the model, run the following command:
```bash
python ./model/regressor.py
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
docker run -p 9000:9000 iris-classifier
```
This will build the Docker image and run the container on port 9000.
