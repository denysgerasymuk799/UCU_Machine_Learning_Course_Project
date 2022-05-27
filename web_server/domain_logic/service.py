import numpy as np
import pandas as pd
import datapane as dp
import plotly.graph_objects as go

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

from domain_logic.constants import *
from init_config import multiplicative_decomposed_trend, multiplicative_decomposed_seasonal, N_FORECASTED_PERIODS


def find_nth(haystack, needle, n):
    start = haystack.find(needle)
    while start >= 0 and n > 1:
        start = haystack.find(needle, start+len(needle))
        n -= 1
    return start


def get_extrapolated_trend(trend):
    X = trend.index[:, np.newaxis]
    y = trend.values[:, np.newaxis]

    poly_reg = PolynomialFeatures(degree=3)
    X_poly = poly_reg.fit_transform(X)
    pol_reg = LinearRegression()
    pol_reg.fit(X_poly, y)

    predict_idx = np.array(
        [trend.index[-1] + i for i in range(1, DECOMPOSED_SHIFT + N_FORECASTED_PERIODS * FORECAST_PERIOD + 1)])
    X_actual_and_forecast = np.concatenate((X, predict_idx[:, np.newaxis]))
    polynomial_trend_prediction = pol_reg.predict(poly_reg.fit_transform(X_actual_and_forecast))

    return polynomial_trend_prediction.flatten()


def extrapolate_seasonality(seasonality, num_periods):
    new_seasonality = pd.concat([seasonality, seasonality[:num_periods * 24]])
    new_seasonality.index = range(0, len(new_seasonality))
    return new_seasonality


def create_feature_df_for_delta(df, df_scaler):
    """
    Create dataframe of feature to train models

    :return: normalized feature numpy arrays and scaler instance which was used for normalizations,
        in the future steps it will be used to return to original values from normalized
    """
    feature_lags = [24 * i for i in range(1, 4)]
    radiation_lags = [24 * i for i in range(1, 4)]
    recent_radiation_lags = [i for i in range(1, 6)]
    feature_df = process_delta_data(df, feature_lags, radiation_lags, recent_radiation_lags)
    feature_df.reset_index(drop=True, inplace=True)
    technical_df = pd.Series([1 for _ in range(feature_df.shape[0])])
    full_df = pd.concat([feature_df, technical_df], axis=1)
    scaled_full_df = df_scaler.transform(full_df)

    return scaled_full_df[:, :-1]


def reshape_for_model(model_name, dataset):
    if 'LSTM' in model_name or \
            'RNN' in model_name or \
            'Conv1d' in model_name:

        if 'LSTM' in model_name:
            # reshape input to be 3D [samples, features, timesteps]
            dataset = dataset.reshape((dataset.shape[0], 1, dataset.shape[1]))

        else:
            # reshape input to be 3D [samples, timesteps, features]
            dataset = dataset.reshape((dataset.shape[0], dataset.shape[1], 1))

    return dataset


def reshape_test_set_for_model(model_name, test_X):
    if 'LSTM' in model_name or \
            'RNN' in model_name or \
            'Conv1d' in model_name:

        if 'LSTM' in model_name:
            test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
        else:
            test_X = test_X.reshape((test_X.shape[0], test_X.shape[1]))

    return test_X


def test_model_with_transform(model, test_X, model_name, first_row_idx):
    """

    Use previous predicted value to construct our feature dataframe and use it for next value prediction
    """
    yhat = []

    test_X = reshape_test_set_for_model(model_name, test_X)

    # Take first row for which we start to forecast.
    # Note that first_row_idx param can be also not equal to zero,
    # for example, it can also be the last index in test_X
    test_row = test_X[first_row_idx]
    prod_lags = test_row[-len(RECENT_RADIATION_LAGS):]
    test_row = test_row.reshape((1, len(test_row)))
    test_row = reshape_for_model(model_name, test_row)
    for i in range(len(test_X)):
        yhat_one_pred = model.predict(test_row)
        yhat.append(yhat_one_pred[0])

        test_row = test_X[i]

        prod_lags = np.roll(prod_lags, 1)
        prod_lags[0] = yhat_one_pred
        new_test_row = np.concatenate((test_row[:-len(RECENT_RADIATION_LAGS)], prod_lags), axis=0)
        test_row = new_test_row.reshape((1, len(new_test_row)))
        test_row = reshape_for_model(model_name, test_row)

    yhat = np.array(yhat)
    yhat[yhat < 0] = 0
    return yhat


def process_delta_data(df, features_lags, radiation_lags, recent_radiation_lags):
    """
    Create a feature dataframe, which will be used for training ML and DL models

    :return: tuple, where first element is dataframe of features filtered from NaN rows, which we got as result of lags for different dataframe columns;
                          second element is a timeseries filtered from NaN entries
    """
    features_df = df.copy()
    features_df.drop(['Year', 'Day',
                      'Month',  # we do not have data for the whole year, hence we need to drop 'Month' column
                      'Hour', 'Log_Radiation'], axis=1, inplace=True, errors='ignore')
    if 'Hourly_DateTime' in features_df.columns:
        features_df.drop(['Hourly_DateTime'], axis=1, inplace=True)

    # Choose features which has good correlation or good logic causation for solar radiation
    feature_columns = ['Temperature', 'Pressure', 'Humidity', 'ZenithDeviation',
                       'WindDirection(Degrees)', 'Speed']

    for feature_column_name in feature_columns:
        for lag in features_lags:
            temp = np.concatenate(
                (np.array([np.nan for _ in range(lag)]), features_df[feature_column_name].values[:-lag]))
            features_df[f'{feature_column_name}_lag_{lag}'] = temp

    # Take radiation lags as one of the features
    # Notice that here lags are more than our target forecast period
    for lag in radiation_lags:
        temp = np.concatenate((np.array([np.nan for _ in range(lag)]), features_df['Radiation'].values[:-lag]))
        features_df[f'Radiation_lag_{lag}'] = temp

    # Take last values, which we forecasted, and use them as features also
    for lag in recent_radiation_lags:
        temp = np.concatenate((np.array([np.nan for _ in range(lag)]), features_df['Radiation'].values[:-lag]))
        features_df[f'Radiation_lag_{lag}'] = temp

    features_df.fillna(features_df.mean(), inplace=True)

    # And finally drop rainfalls
    features_df.drop(feature_columns, axis=1, inplace=True)
    features_df.drop(['Radiation'], axis=1, inplace=True)

    return features_df[max(features_lags + radiation_lags):]


def predict_out_of_df(test_model, df_scaler, original_df, test_df, input_features_df,
                      forecast_datetime_range, model_name):
    full_df_X = create_feature_df_for_delta(test_df, df_scaler)

    # reshape input to be 3D [samples, timestamps, features]
    full_df_X = full_df_X.reshape((full_df_X.shape[0], full_df_X.shape[1], 1))

    yhat = test_model_with_transform(test_model, full_df_X, model_name, -1)
    full_df_X = full_df_X.reshape((full_df_X.shape[0], full_df_X.shape[1]))

    # invert scaling for forecast
    inv_yhat = np.concatenate((full_df_X, yhat), axis=1)
    inv_yhat = df_scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:, -1]

    predicted_seasonality = extrapolate_seasonality(multiplicative_decomposed_seasonal, 3)[-FORECAST_PERIOD:]
    predicted_trend = get_extrapolated_trend(multiplicative_decomposed_trend)[-FORECAST_PERIOD:]

    model_prediction_initial_series = inv_yhat * predicted_trend * predicted_seasonality

    start_idx = original_df.shape[0]
    end_idx = start_idx + FORECAST_PERIOD

    before_lag = 14 * 24
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=original_df['Hourly_DateTime'][start_idx + 1 - before_lag: start_idx + 1],  # add + 1 to make consistent plot
        y=original_df['Radiation'][start_idx + 1 - before_lag: start_idx + 1],
        name='Last days original radiation',  # Style name/legend entry with html tags
    ))
    fig.add_trace(go.Scatter(
        x=forecast_datetime_range,
        y=model_prediction_initial_series,
        name='Forecast',
    ))

    results = {
        'Date and Time': forecast_datetime_range,
        'Forecast': model_prediction_initial_series.reset_index(drop=True)
    }
    result_df = pd.DataFrame(results)

    statistics_df = result_df.agg(
        {
            "Forecast": ["sum", "min", "max", "mean"],
        }
    )

    dp.Report("# Solar Radiation Forecast Report",
              "## Input features for the last 3 days (top 10 rows)",
              dp.Table(input_features_df[:10], caption="Input features"),
              "## Forecast for 3 days ahead",
              dp.Plot(fig, caption="Forecast for 3 days ahead"),
              "## Forecast statistics",
              dp.Table(statistics_df, caption="Dataframe of forecast statistics"),
              "## Dataframe of forecast values",
              dp.Table(result_df, caption="Dataframe of forecast values")
              ).save(path='./results/report.html')
    # print('report_html -- ', report_html._gen_report(embedded=False, title='Model title'))
    #
    # with open(HOME_DIR + '/results/report.html', 'w') as html_file:
    #     html_file.write(report_html.__str__())
    # ).save(path=HOME_DIR + '/results/report.html', open=True)
