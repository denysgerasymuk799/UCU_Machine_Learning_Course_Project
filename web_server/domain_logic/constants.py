import os

FORECAST_DAYS = 3
FORECAST_PERIOD = 24 * FORECAST_DAYS # 3 days
DECOMPOSED_SHIFT = FORECAST_PERIOD // 2 + (FORECAST_PERIOD // 2) % 24  # round to the closest number of days

FEATURES_LAGS = [24 * 2 + 24 * i for i in range(1, 4)]
RADIATION_LAGS = [24 * 2 + 24 * i for i in range(1, 4)]
RECENT_RADIATION_LAGS = [i for i in range(1, 6)]
MAX_DF_SHIFT = max(FEATURES_LAGS + RADIATION_LAGS)
# HOME_DIR = '/content/drive/MyDrive/Colab Notebooks/UCU_ML_2022/UCU_Machine_Learning_Course_Project'
# HOME_DIR = '/home/denys_herasymuk/UCU/3course_2term/Machine_Learning/Course_Project/UCU_Machine_Learning_Course_Project'
HOME_DIR = os.getcwd()
