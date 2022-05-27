import pandas as pd

from datetime import timedelta
from fastapi import FastAPI, Request, Response
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from web_server.domain_logic.constants import *
from web_server.domain_logic.service import predict_out_of_df
from init_config import reconstructed_model, delta_scaler, N_FORECASTED_PERIODS, hourly_radiation_df

app = FastAPI()


class Item(BaseModel):
    delta_features_path: str


@app.get("/forecast_report")
async def get_forecast_report():
    # with open(HOME_DIR + '/results/report.html', 'r') as html_file:
    with open('./report.html', 'r') as html_file:
        content = html_file.read()

    return HTMLResponse(content=content, status_code=200)


@app.post("/load_features")
async def load_features(request: Item):
    global hourly_radiation_df, N_FORECASTED_PERIODS

    delta_features_path = request.delta_features_path

    # Load feature df for the last 3 days.
    # Note that Radiation is one of columns in delta_features,
    # since we use values of Radiation for the last 3 days as a feature
    delta_features = pd.read_csv(delta_features_path)
    delta_features['Hourly_DateTime'] = pd.to_datetime(delta_features['Hourly_DateTime'])
    forecast_datetime_range = delta_features['Hourly_DateTime'] + timedelta(days=FORECAST_DAYS)

    hourly_radiation_df = pd.concat([hourly_radiation_df, delta_features], axis=0)

    delta_features_extended = pd.concat([delta_features, delta_features])
    predict_out_of_df(reconstructed_model, delta_scaler, hourly_radiation_df,
                      delta_features_extended, delta_features,
                      forecast_datetime_range, model_name='Conv1d')
    N_FORECASTED_PERIODS += 1
    return Response(status_code=200, content='Features loaded successfully')


@app.get("/reset")
async def reset_variables():
    global multiplicative_decomposed_trend, multiplicative_decomposed_seasonal,\
        hourly_radiation_df, N_FORECASTED_PERIODS

    N_FORECASTED_PERIODS = 1

    # Import trend and seasonality to use them during forecasting
    multiplicative_decomposed_trend = pd.read_csv(HOME_DIR + '/results/multiplicative_decomposed_trend_v1.csv',
                                                  header=0, index_col=0, squeeze=True)
    multiplicative_decomposed_seasonal = pd.read_csv(HOME_DIR + '/results/multiplicative_decomposed_seasonal_v1.csv',
                                                     header=0, index_col=0, squeeze=True)

    # For Google Colab here is path to dataset on Google Drive
    hourly_radiation_df = pd.read_csv(HOME_DIR + '/data/dataset1_HourlySolarRadiationProcessed_s3.csv')
    # hourly_radiation_df = pd.read_csv(os.path.join("..", "data", "dataset1_HourlySolarRadiationProcessed.csv"))
    hourly_radiation_df['Hourly_DateTime'] = pd.to_datetime(hourly_radiation_df['Hourly_DateTime'])


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, port=8091)
