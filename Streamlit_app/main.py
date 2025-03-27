import streamlit as st
import pandas as pd


import joblib

st.set_page_config(page_title="Прогноз цен на арматуру", layout="wide")

st.title(" Прогноз цен на арматуру для менеджера")

@st.cache_resource
def load_model():
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

@st.cache_data
def load_data():
    return pd.read_excel("Data.xlsx", engine="openpyxl")

def create_features(Data):
    Data = Data.copy()
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
        "Цена на арматуру": Data["Цена на арматуру"].max()
    }
    Data = pd.concat([Data, pd.DataFrame([new_row])], ignore_index=True)

    future_data = []

    for i in range(horizon_weeks):
        new_date = Data.iloc[-1]["dt"] + pd.Timedelta(weeks=1)
        new_row = {
            "dt": new_date,
            "Цена на арматуру": Data["Цена на арматуру"].iloc[-1]
        }
        Data = pd.concat([Data, pd.DataFrame([new_row])], ignore_index=True)
        Data = create_features(Data)

        if Data[features].shape[0] == 0:
            break

        X_pred = scaler.transform(Data[features].iloc[[-1]])
        pred = model.predict(X_pred)[0]

        Data.loc[Data.index[-1], "Цена на арматуру"] = pred

        future_data.append({
            "dt": new_date,
            "Прогноз": pred
        })

    return pd.DataFrame(future_data)


st.sidebar.header(" Данные")
uploaded_file = st.sidebar.file_uploader("Загрузить новые данные .xlsx ", type="xlsx")

if uploaded_file:
    Data = pd.read_excel(uploaded_file, engine="openpyxl")
    st.sidebar.success("Загружен пользовательский файл")
else:
    Data = load_data()
    st.sidebar.info(" Используются стандартные данные")

st.sidebar.header(" Параметры")
horizon = st.sidebar.selectbox("Горизонт прогноза (недель):", [4, 12, 26, 52, 56, 64, 104])

if st.sidebar.button("Построить прогноз"):
    model, scaler = load_model()
    df_prepared = create_features(Data)
    features = [col for col in df_prepared.columns if col not in ["dt", "Цена на арматуру"]]

    forecast_df = forecast(Data, model, scaler, features, horizon_weeks=horizon)
    min_row = forecast_df.loc[forecast_df['Прогноз'].idxmin()]

    st.subheader(" Прогноз на следующие недели")
    st.line_chart(forecast_df.set_index("dt")["Прогноз"])

    st.success(f" Минимальная цена: **{int(min_row['Прогноз'])} ₽** — дата покупки: **{min_row['dt'].date()}**")
    st.dataframe(forecast_df.set_index("dt"))

