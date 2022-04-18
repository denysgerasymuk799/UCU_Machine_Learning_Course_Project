FORECAST_PERIOD = 24

FEATURES_LAGS = [24 * i for i in range(1, 6)]
RADIATION_LAGS = [24 * i for i in range(1, 6)]
RECENT_RADIATION_LAGS = [i for i in range(1, 6)]
MAX_DF_SHIFT = max(FEATURES_LAGS + RADIATION_LAGS)