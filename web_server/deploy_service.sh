docker build . -t forecast_service:1.0

docker tag forecast_service:1.0 218145147595.dkr.ecr.eu-central-1.amazonaws.com/ml_project_2022

docker push 218145147595.dkr.ecr.eu-central-1.amazonaws.com/ml_project_2022