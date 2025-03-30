import streamlit as st
import pandas as pd
import numpy as np
import joblib
import io

st.set_page_config(page_title="Прогноз цен на арматуру", layout="wide")
st.title("📈 Прогноз цен на арматуру для менеджера")

@st.cache_resource
def load_model():
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

@st.cache_data
def load_data():
    return pd.read_excel("Data.xlsx", engine="openpyxl")

def outliers(Data, col="Цена на арматуру", window=4, threshold=2.5):
    Data = Data.copy()
    Data["MA"] = Data[col].rolling(window=window, center=True).mean()
    Data["delta"] = np.abs(Data[col] - Data["MA"])
    std = Data["delta"].std()
    Data["cleaned"] = np.where(Data["delta"] > threshold * std, Data["MA"], Data[col])
    return Data

def create_features(Data):
    Data = Data.copy()
    Data = outliers(Data)
    Data["Цена на арматуру"] = Data["cleaned"]
    Data["dt"] = pd.to_datetime(Data["dt"])
    Data = Data.sort_values("dt")
    Data["week"] = Data["dt"].dt.isocalendar().week
    Data["month"] = Data["dt"].dt.month
    Data["quarter"] = Data["dt"].dt.quarter
    Data["year"] = Data["dt"].dt.year

    for lag in [1, 2, 4, 12]:
        Data[f"lag_{lag}"] = Data["Цена на арматуру"].shift(lag)
        Data[f"rolling_mean_{lag}"] = Data["Цена на арматуру"].shift(1).rolling(window=lag).mean()

    return Data.fillna(Data.median(numeric_only=True)).reset_index(drop=True)

def forecast(Data, model, scaler, features, horizon_weeks=52):
    Data = create_features(Data)
    Data = Data.sort_values("dt")
    today = pd.to_datetime("today").normalize()

    new_row = {
        "dt": today,
        "Цена на арматуру": Data["Цена на арматуру"].iloc[-1]
    }
    Data = pd.concat([Data, pd.DataFrame([new_row])], ignore_index=True)

    future_data = []

    for _ in range(horizon_weeks):
        new_date = Data.iloc[-1]["dt"] + pd.Timedelta(weeks=1)
        new_row = {
            "dt": new_date,
            "Цена на арматуру": Data["Цена на арматуру"].iloc[-1]
        }
        Data = pd.concat([Data, pd.DataFrame([new_row])], ignore_index=True)
        Data = create_features(Data)

        last_row = Data[features].iloc[[-1]]

        if hasattr(scaler, 'feature_names_in_'):
            missing = set(scaler.feature_names_in_) - set(last_row.columns)
            if missing:
                raise ValueError(f"Отсутствующие признаки: {missing}")
            last_row = last_row.loc[:, scaler.feature_names_in_]

        X_pred = scaler.transform(last_row)
        pred = model.predict(X_pred)[0]

        Data.loc[Data.index[-1], "Цена на арматуру"] = pred
        future_data.append({"dt": new_date, "Прогноз": pred})

    return pd.DataFrame(future_data)

# === Интерфейс Streamlit ===
st.sidebar.header("📂 Данные")
uploaded_file = st.sidebar.file_uploader("Загрузить данные (.xlsx)", type="xlsx")

if uploaded_file:
    Data = pd.read_excel(uploaded_file, engine="openpyxl")
    st.sidebar.success("✅ Загружен пользовательский файл")
else:
    Data = load_data()
    st.sidebar.info("ℹ️ Используются стандартные данные")

st.sidebar.header("⚙️ Параметры прогноза")
horizon = st.sidebar.selectbox("Горизонт прогноза (недель):", [4, 12, 26, 52, 56, 64, 104])

if st.sidebar.button("📊 Построить прогноз"):
    model, scaler = load_model()
    df_prepared = create_features(Data)
    features = [col for col in df_prepared.columns if col not in ["dt", "Цена на арматуру"]]

    forecast_df = forecast(Data, model, scaler, features, horizon_weeks=horizon)
    min_row = forecast_df.loc[forecast_df['Прогноз'].idxmin()]

    st.subheader("📅 Прогноз на будущие недели")
    st.line_chart(forecast_df.set_index("dt")["Прогноз"])

    st.success(f"💰 Минимальная цена: **{int(min_row['Прогноз'])} ₽** — 📅 дата: **{min_row['dt'].strftime('%d.%m.%Y')}**")
    st.dataframe(forecast_df.set_index("dt"))

    # === РЕКОМЕНДАЦИЯ ПО ЗАКУПКЕ ===
    forecast_series = forecast_df["Прогноз"].values
    buy_weeks = horizon  # по умолчанию — весь горизонт

    for i in range(1, len(forecast_series) - 1):
        if forecast_series[i] < forecast_series[i - 1] and forecast_series[i] < forecast_series[i + 1]:
            buy_weeks = i + 1
            break

    buy_date = forecast_df.iloc[buy_weeks - 1]["dt"]
    st.info(f"🛒 **Рекомендация:** Закупить арматуру на **{buy_weeks} недель(и)** вперёд до **{buy_date.strftime('%d.%m.%Y')}**, чтобы избежать повышения цен.")

    # === КНОПКА СКАЧИВАНИЯ ===
    buffer = io.BytesIO
