import joblib
import pandas as pd

from keras.models import load_model

from web_server.domain_logic.constants import *


N_FORECASTED_PERIODS = 1

# Import the custom model
dir_path = HOME_DIR + '/results/'  # for Google Colab
# dir_path = os.getcwd() + '/'
reconstructed_model = load_model(dir_path + 'custom_model_v3.h5')

# Load MinMaxScaler.
# Since we used it on training, validation and testing datasets, it is better also use the same on the new coming data
delta_scaler = joblib.load(HOME_DIR + '/results/df_scaler_v3.pkl')

# Import trend and seasonality to use them during forecasting
multiplicative_decomposed_trend = pd.read_csv(HOME_DIR + '/results/multiplicative_decomposed_trend_v1.csv',
                                              header=0, index_col=0, squeeze=True)
multiplicative_decomposed_seasonal = pd.read_csv(HOME_DIR + '/results/multiplicative_decomposed_seasonal_v1.csv',
                                                 header=0, index_col=0, squeeze=True)

# For Google Colab here is path to dataset on Google Drive
hourly_radiation_df = pd.read_csv(HOME_DIR + '/data/dataset1_HourlySolarRadiationProcessed_s3.csv')
# hourly_radiation_df = pd.read_csv(os.path.join("..", "data", "dataset1_HourlySolarRadiationProcessed.csv"))
hourly_radiation_df['Hourly_DateTime'] = pd.to_datetime(hourly_radiation_df['Hourly_DateTime'])
