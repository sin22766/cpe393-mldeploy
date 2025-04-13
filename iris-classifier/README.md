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
docker run -p 9000:9000 iris-classifier
```
This will build the Docker image and run the container on port 9000.

## Sample Request
To test the model, you can use the following curl command:
```bash
curl --request POST \
  --url http://localhost:9000/predict \
  --header 'content-type: application/json' \
  --data '{
  "features": [
    {
      "area": 8372,
      "bedrooms": 3,
      "bathrooms": 1,
      "stories": 3,
      "mainroad": true,
      "guestroom": false,
      "basement": false,
      "hotwaterheating": false,
      "airconditioning": true,
      "parking": 2,
      "prefarea": false,
      "furnishingstatus": 0
    }
  ]
}'
```
This will send a POST request to the `/predict` endpoint with the features of the house. The response will be a JSON object with the predicted price of the house.