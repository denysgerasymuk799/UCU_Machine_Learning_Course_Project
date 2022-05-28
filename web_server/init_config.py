import os
import io
import boto3
import joblib
import pandas as pd

from dotenv import load_dotenv
from keras.models import load_model

from domain_logic.constants import *

N_FORECASTED_PERIODS = 1
load_dotenv(dotenv_path='./web_server.env')

# Accessing the S3 buckets using boto3 client
s3_client = boto3.client('s3',
                         aws_access_key_id=os.getenv('ACCESS_KEY'),
                         aws_secret_access_key=os.getenv('SECRET_ACCESS_KEY'),
                         region_name=os.getenv('REGION_NAME')
                         )
s3_bucket_name = os.getenv('BUCKET_NAME')

# Import the custom model
os.makedirs('results', exist_ok=True)
model_name = 'custom_model_v3.h5'
dir_path = HOME_DIR + '/results/'
s3_client.download_file(s3_bucket_name, f'results/{model_name}', dir_path + model_name)
reconstructed_model = load_model(dir_path + model_name)

# Load MinMaxScaler.
# Since we used it on training, validation and testing datasets, it is better also use the same on the new coming data
s3_client.download_file(s3_bucket_name, 'results/df_scaler_v3.pkl', HOME_DIR + '/results/df_scaler_v3.pkl')
delta_scaler = joblib.load(HOME_DIR + '/results/df_scaler_v3.pkl')

# Import trend and seasonality to use them during forecasting
trend_obj = s3_client.get_object(Bucket=s3_bucket_name, Key='results/multiplicative_decomposed_trend_v1.csv')
multiplicative_decomposed_trend = pd.read_csv(io.BytesIO(trend_obj['Body'].read()),
                                              header=0, index_col=0, squeeze=True)
seasonal_obj = s3_client.get_object(Bucket=s3_bucket_name, Key='results/multiplicative_decomposed_seasonal_v1.csv')
multiplicative_decomposed_seasonal = pd.read_csv(io.BytesIO(seasonal_obj['Body'].read()),
                                                 header=0, index_col=0, squeeze=True)

hourly_radiation_obj = s3_client.get_object(Bucket=s3_bucket_name, Key='data/dataset1_HourlySolarRadiationProcessed_s3.csv')
hourly_radiation_df = pd.read_csv(io.BytesIO(hourly_radiation_obj['Body'].read()))
hourly_radiation_df['Hourly_DateTime'] = pd.to_datetime(hourly_radiation_df['Hourly_DateTime'])
