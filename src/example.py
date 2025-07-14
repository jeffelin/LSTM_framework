from scalecast.Forecaster import Forecaster
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# using sunspots.csv 

DEBUG = False

ALL_MODEL_NAMES = ['rnn_1layer', 'rnn_layered', 'lstm_1layer', 'lstm_layered', 'nnar']


def setup_forecaster():
    df = pd.read_csv('Sunspots.csv', parse_dates=['Date'])
    df.rename(columns={'Monthly Mean Total Sunspot Number': 'Target'}, inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date').asfreq('M').reset_index()

    if DEBUG:
        print(df.head())

    return Forecaster(
        y=df['Target'],
        current_dates=df['Date'],
        test_length=240,
        future_dates=240,
        cis=True,
    )


def eda(f):
    f.plot()
    plt.title('Original Sunspots Series', size=16)
    plt.tight_layout()
    plt.show()

    f.plot_acf(lags=240)
    plt.tight_layout()
    plt.show()

    f.plot_pacf(lags=240, method='ywm')
    plt.tight_layout()
    plt.show()

    f.seasonal_decompose(period=12).plot()
    plt.tight_layout()
    plt.show()


def forecast_rnn_simple(f):
    f.auto_Xvar_select(
        try_trend=False,
        irr_cycles=[120, 132, 144],
        cross_validate=True,
        cvkwargs={'k': 3},
        dynamic_tuning=240,
    )
    f.set_estimator('rnn')
    f.manual_forecast(
        layers_struct=[('SimpleRNN', {'units': 100, 'dropout': 0.2})],
        epochs=25,
        validation_split=0.2,
        plot_loss=False,
        call_me="rnn_1layer",
        verbose=0,
    )


def forecast_rnn_layered(f):
    f.manual_forecast(
        layers_struct=[('SimpleRNN', {'units': 100, 'dropout': 0})] * 2 +
                      [('Dense', {'units': 10})] * 2,
        epochs=25,
        validation_split=0.2,
        call_me='rnn_layered',
        verbose=0,
    )


def forecast_lstm_simple(f):
    f.manual_forecast(
        layers_struct=[('LSTM', {'units': 100, 'dropout': 0.2})],
        epochs=15,
        validation_split=0.2,
        plot_loss=False,
        call_me="lstm_1layer",
        verbose=0,
    )


def forecast_lstm_layered(f):
    f.manual_forecast(
        layers_struct=[('LSTM', {'units': 100, 'dropout': 0})] * 2 +
                      [('Dense', {'units': 10})] * 2,
        epochs=15,
        validation_split=0.2,
        random_seed=42,
        plot_loss=False,
        call_me='lstm_layered',
        verbose=0,
    )


def prepare_nnar_model(f):
    f.drop_all_Xvars()
    f.add_ar_terms(10)
    f.add_AR_terms((6, 12))
    f.add_seasonal_regressors('month', raw=False, sincos=True)
    f.add_cycle(120)


def forecast_nnar(f):
    k = int(np.ceil(len(f.get_regressor_names()) / 2))
    f.set_estimator('mlp')
    for r in tqdm(range(20), desc='Training MLPs'):
        try:
            f.manual_forecast(
                hidden_layer_sizes=(k,),
                activation='relu',
                random_state=r,
                normalizer='scale',
                call_me=f'mlp_{r}',
            )
        except Exception as e:
            if DEBUG:
                print(f"❌ MLP model mlp_{r} failed: {e}")

    f.save_feature_importance()
    f.set_estimator('combo')
    f.manual_forecast(
        how='simple',
        models=[f'mlp_{r}' for r in range(20) if f.model_exists(f'mlp_{r}')],
        call_me='nnar'
    )


def compare_all_models(f):
    available_models = [m for m in ALL_MODEL_NAMES if m in f.get_models()]

    if not available_models:
        print("No models available for comparison.")
        return

    # Test set comparison plot
    f.plot_test_set(
        models=available_models,
        order_by='TestSetRMSE',
        include_train=False,
        ci=True,
    )
    plt.title('All Models Test Performance - Test Set Only', size=16)
    plt.tight_layout()
    plt.show()

    # Forecast plot for top 3 models
    top_models = [m for m in ['rnn_layered', 'lstm_layered', 'nnar'] if m in available_models]
    f.plot(models=top_models, order_by='TestSetRMSE', ci=True)
    plt.title('Top-3 Model Forecasts - Next 20 Years', size=16)
    plt.tight_layout()
    plt.show()

    # Summary export
    summary = f.export(
        'model_summaries',
        models=available_models,
        determine_best_by='TestSetRMSE'
    )[[  # Select relevant columns
        'ModelNickname',
        'TestSetRMSE',
        'TestSetR2',
        'InSampleRMSE',
        'InSampleR2',
        'best_model'
    ]]
    print(summary)


def plot_actual_and_forecast(f, model_name='lstm_layered'):
    if model_name not in f.forecast_df.columns:
        print(f"⚠️ Model '{model_name}' not found in forecast_df.")
        return

    plt.figure(figsize=(12, 6))
    plt.plot(f.current_dates, f.y, label='Actual', color='blue')
    plt.plot(f.forecast_df.index, f.forecast_df[model_name], label='Forecast', color='orange')
    plt.title(f'Actual Sunspots with Forecast ({model_name})', fontsize=16)
    plt.xlabel('Date')
    plt.ylabel('Sunspots')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def run_all():
    f = setup_forecaster()
    eda(f)
    forecast_rnn_simple(f)
    forecast_rnn_layered(f)
    forecast_lstm_simple(f)
    forecast_lstm_layered(f)
    prepare_nnar_model(f)
    forecast_nnar(f)
    compare_all_models(f)
    plot_actual_and_forecast(f, model_name='lstm_layered')


if __name__ == '__main__':
    run_all()
