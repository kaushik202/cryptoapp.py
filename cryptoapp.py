import streamlit as st
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import joblib
import pmdarima as pm
from statsmodels.tsa.stattools import adfuller
from prophet import Prophet
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Set random seeds for reproducibility
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ['TF_DETERMINISTIC_OPS'] = '1'

st.set_page_config(page_title="Crypto Price Forecasting", layout="wide")
st.title("🚀 Cryptocurrency Price Forecasting with ML")
st.markdown("Compare ARIMA, Prophet, and LSTM models for time series forecasting")

# Sidebar configuration
st.sidebar.header("⚙️ Configuration")
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=['csv'])
test_size_pct = st.sidebar.slider("Test Size (%)", 5, 20, 10)
val_size_pct = st.sidebar.slider("Validation Size (%)", 5, 20, 10)
n_steps = st.sidebar.slider("LSTM Lookback Period (days)", 30, 90, 60)
future_days = st.sidebar.slider("Future Forecast Days", 30, 90, 60)

# Helper functions
def create_sequences_from_series(series_pd, n_steps):
    x, y, idx = [], [], []
    arr = series_pd.values
    for i in range(n_steps, len(series_pd)):
        x.append(arr[i - n_steps : i])
        y.append(arr[i])
        idx.append(series_pd.index[i])
    x = np.array(x)
    y = np.array(y).reshape(-1)
    idx = pd.DatetimeIndex(idx)
    return x, y, idx

def make_lstm_model(n_steps):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(n_steps, 1)),
        Dropout(0.2),
        LSTM(32),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model

# Main app logic
if uploaded_file is not None:
    # Load and prepare data
    with st.spinner("Loading data..."):
        df = pd.read_csv(uploaded_file, parse_dates=["snapped_at"])
        df = df.set_index("snapped_at").sort_index()
        df.index = df.index.tz_convert(None) if df.index.tz is not None else df.index
        df = df.resample("1D").ffill()
        series = df["price"].astype(float)
    
    st.success(f"✅ Data loaded: {len(series)} data points from {series.index.min().date()} to {series.index.max().date()}")
    
    # Display data overview
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Data Points", len(series))
    with col2:
        st.metric("Min Price", f"${series.min():.2f}")
    with col3:
        st.metric("Max Price", f"${series.max():.2f}")
    
    # Plot raw data
    st.subheader("📈 Historical Price Data")
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(series.index, series.values, linewidth=1)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price ($)")
    ax.set_title("Historical Cryptocurrency Prices")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    
    # Train/Test/Validation split
    n = len(series)
    test_size = int(test_size_pct / 100 * n)
    val_size = int(val_size_pct / 100 * n)
    
    train = series.iloc[: n - val_size - test_size]
    val = series.iloc[n - val_size - test_size : n - test_size]
    test = series.iloc[n - test_size :]
    
    st.info(f"📊 Data Split - Train: {len(train)} | Validation: {len(val)} | Test: {len(test)}")
    
    # Model training section
    st.header("🤖 Model Training & Evaluation")
    
    if st.button("🚀 Train All Models", type="primary"):
        results = {}
        
        # ARIMA Model
        with st.spinner("Training ARIMA model..."):
            st.subheader("1️⃣ ARIMA Model")
            adf_result = adfuller(train.dropna())
            st.write(f"**ADF Test p-value:** {adf_result[1]:.4f} {'(Non-stationary)' if adf_result[1] > 0.05 else '(Stationary)'}")
            
            arima_model = pm.auto_arima(
                train,
                start_p=0, start_q=0,
                max_p=5, max_q=5,
                seasonal=False,
                trace=False,
                error_action="ignore",
                suppress_warnings=True,
                stepwise=True,
                random_state=SEED
            )
            
            val_pred_arima = arima_model.predict(n_periods=len(val))
            mae_arima = mean_absolute_error(val, val_pred_arima)
            rmse_arima = np.sqrt(mean_squared_error(val, val_pred_arima))
            results['ARIMA'] = {'MAE': mae_arima, 'RMSE': rmse_arima}
            st.success(f"✅ ARIMA - MAE: ${mae_arima:.2f}, RMSE: ${rmse_arima:.2f}")
        
        # Prophet Model
        with st.spinner("Training Prophet model..."):
            st.subheader("2️⃣ Prophet Model")
            df_prophet = train.reset_index().rename(columns={"snapped_at": "ds", "price": "y"})
            df_prophet["ds"] = pd.to_datetime(df_prophet["ds"]).dt.tz_localize(None)
            
            m = Prophet(daily_seasonality=True)
            m.fit(df_prophet)
            
            future = m.make_future_dataframe(periods=len(val), freq="D")
            forecast = m.predict(future)
            val_pred_prophet = forecast["yhat"].iloc[-len(val):].values
            
            mae_prophet = mean_absolute_error(val, val_pred_prophet)
            rmse_prophet = np.sqrt(mean_squared_error(val, val_pred_prophet))
            results['Prophet'] = {'MAE': mae_prophet, 'RMSE': rmse_prophet}
            st.success(f"✅ Prophet - MAE: ${mae_prophet:.2f}, RMSE: ${rmse_prophet:.2f}")
        
        # LSTM Model
        with st.spinner("Training LSTM model (this may take a few minutes)..."):
            st.subheader("3️⃣ LSTM Model")
            
            # Prepare scaled data
            scaler = MinMaxScaler()
            train_scaled = scaler.fit_transform(train.values.reshape(-1, 1))
            val_scaled = scaler.transform(val.values.reshape(-1, 1))
            test_scaled = scaler.transform(test.values.reshape(-1, 1))
            
            # Create training sequences
            train_pd = pd.Series(train_scaled.flatten(), index=train.index)
            x_train_seq, y_train_seq, idx_train = create_sequences_from_series(train_pd, n_steps)
            x_train = x_train_seq.reshape((x_train_seq.shape[0], x_train_seq.shape[1], 1))
            y_train = y_train_seq
            
            # Create validation sequences
            concat_tv = pd.concat([train.tail(n_steps), val])
            concat_tv_pd = pd.Series(scaler.transform(concat_tv.values.reshape(-1,1)).flatten(), index=concat_tv.index)
            x_val_seq_all, y_val_seq_all, idx_val_all = create_sequences_from_series(concat_tv_pd, n_steps)
            
            mask = idx_val_all.isin(val.index)
            x_val_seq = x_val_seq_all[mask]
            y_val_seq = y_val_seq_all[mask]
            idx_val = idx_val_all[mask]
            
            x_val = x_val_seq.reshape((x_val_seq.shape[0], x_val_seq.shape[1], 1))
            y_val = y_val_seq
            
            # Build and train LSTM
            model = make_lstm_model(n_steps)
            es = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            class StreamlitCallback(tf.keras.callbacks.Callback):
                def on_epoch_end(self, epoch, logs=None):
                    progress = min((epoch + 1) / 100, 1.0)
                    progress_bar.progress(progress)
                    status_text.text(f"Epoch {epoch + 1}/100 - Loss: {logs['loss']:.4f}, Val Loss: {logs['val_loss']:.4f}")
            
            history = model.fit(
                x_train, y_train,
                validation_data=(x_val, y_val),
                epochs=100,
                batch_size=32,
                callbacks=[es, StreamlitCallback()],
                shuffle=False,
                verbose=0
            )
            
            progress_bar.empty()
            status_text.empty()
            
            # Validation predictions
            pred_val_scaled = model.predict(x_val, verbose=0)
            pred_val = scaler.inverse_transform(pred_val_scaled).flatten()
            val_true = val.loc[idx_val].values
            
            mae_lstm = mean_absolute_error(val_true, pred_val)
            rmse_lstm = np.sqrt(mean_squared_error(val_true, pred_val))
            results['LSTM'] = {'MAE': mae_lstm, 'RMSE': rmse_lstm}
            st.success(f"✅ LSTM - MAE: ${mae_lstm:.2f}, RMSE: ${rmse_lstm:.2f}")
            
            # Retrain on full data (train + val)
            st.info("🔄 Retraining LSTM on combined train+validation data...")
            full_train = pd.concat([train, val])
            full_scaled = scaler.fit_transform(full_train.values.reshape(-1, 1))
            full_pd = pd.Series(full_scaled.flatten(), index=full_train.index)
            
            x_full_seq, y_full_seq, idx_full = create_sequences_from_series(full_pd, n_steps)
            x_full = x_full_seq.reshape((x_full_seq.shape[0], x_full_seq.shape[1], 1))
            y_full = y_full_seq
            
            final_model = make_lstm_model(n_steps)
            final_es = EarlyStopping(monitor="loss", patience=10, restore_best_weights=True)
            final_model.fit(x_full, y_full, epochs=100, batch_size=32, callbacks=[final_es], shuffle=False, verbose=0)
            
            # Test set evaluation
            concat_ft = pd.concat([full_train.tail(n_steps), test])
            concat_ft_pd = pd.Series(scaler.transform(concat_ft.values.reshape(-1,1)).flatten(), index=concat_ft.index)
            x_test_all, y_test_all, idx_test_all = create_sequences_from_series(concat_ft_pd, n_steps)
            
            mask_test = idx_test_all.isin(test.index)
            x_test_seq = x_test_all[mask_test]
            idx_test = idx_test_all[mask_test]
            
            x_test = x_test_seq.reshape((x_test_seq.shape[0], x_test_seq.shape[1], 1))
            pred_test_scaled = final_model.predict(x_test, verbose=0)
            pred_test = scaler.inverse_transform(pred_test_scaled).flatten()
            y_test_true = test.loc[idx_test].values
            
            mae_test = mean_absolute_error(y_test_true, pred_test)
            rmse_test = np.sqrt(mean_squared_error(y_test_true, pred_test))
            
            st.success(f"🎯 **Final LSTM Test Performance** - MAE: ${mae_test:.2f}, RMSE: ${rmse_test:.2f}")
            
            # Future forecasting
            st.info(f"🔮 Generating {future_days}-day forecast...")
            last_seq = full_pd.values[-n_steps:].reshape(1, n_steps, 1)
            
            future_preds_scaled = []
            cur_seq = last_seq.copy()
            for _ in range(future_days):
                p = final_model.predict(cur_seq, verbose=0)[0][0]
                future_preds_scaled.append(p)
                cur_seq = np.append(cur_seq[:, 1:, :], [[[p]]], axis=1)
            
            future_preds = scaler.inverse_transform(np.array(future_preds_scaled).reshape(-1,1)).flatten()
            future_index = pd.date_range(start=series.index.max() + pd.Timedelta(days=1), periods=future_days, freq="D")
            future_df = pd.Series(future_preds, index=future_index)
            
            # Store in session state
            st.session_state['results'] = results
            st.session_state['idx_test'] = idx_test
            st.session_state['y_test_true'] = y_test_true
            st.session_state['pred_test'] = pred_test
            st.session_state['future_df'] = future_df
            st.session_state['series'] = series
        
        # Model comparison
        st.header("📊 Model Comparison")
        comparison_df = pd.DataFrame(results).T
        comparison_df = comparison_df.sort_values('MAE')
        
        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(comparison_df.style.highlight_min(axis=0, color='lightgreen'))
        
        with col2:
            fig, ax = plt.subplots(figsize=(8, 5))
            x = np.arange(len(comparison_df))
            width = 0.35
            ax.bar(x - width/2, comparison_df['MAE'], width, label='MAE', alpha=0.8)
            ax.bar(x + width/2, comparison_df['RMSE'], width, label='RMSE', alpha=0.8)
            ax.set_ylabel('Error ($)')
            ax.set_title('Model Performance Comparison')
            ax.set_xticks(x)
            ax.set_xticklabels(comparison_df.index)
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
        
        # Final visualization
        st.header("📈 Final Predictions & Future Forecast")
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(series.index, series.values, label="Historical Data", alpha=0.6, linewidth=1)
        ax.plot(idx_test, y_test_true, label="Test Actual", color="tab:blue", linewidth=2)
        ax.plot(idx_test, pred_test, label="LSTM Test Prediction", color="tab:orange", linewidth=2)
        ax.plot(future_df.index, future_df.values, label=f"Future Forecast ({future_days} days)", color="tab:green", linewidth=2, linestyle='--')
        ax.set_xlabel("Date")
        ax.set_ylabel("Price ($)")
        ax.set_title("Cryptocurrency Price: Historical, Test Predictions & Future Forecast (LSTM)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        
        # Future forecast table
        st.subheader("🔮 Future Price Predictions")
        forecast_display = future_df.reset_index()
        forecast_display.columns = ['Date', 'Predicted Price ($)']
        forecast_display['Predicted Price ($)'] = forecast_display['Predicted Price ($)'].round(2)
        st.dataframe(forecast_display.head(30), use_container_width=True)

else:
    st.info("👆 Please upload a CSV file with 'snapped_at' (datetime) and 'price' (float) columns to begin.")
    st.markdown("""
    ### 📋 Expected CSV Format:
    - **snapped_at**: Datetime column (e.g., '2024-01-01 00:00:00')
    - **price**: Numeric price values
    
    ### 🎯 Models Included:
    1. **ARIMA**: Traditional statistical model for time series
    2. **Prophet**: Facebook's forecasting model with seasonality detection
    3. **LSTM**: Deep learning model for complex patterns
    """)
