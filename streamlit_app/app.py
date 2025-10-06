import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import Input
import tensorflow as tf
import random
import os

# --- Set Seed Global ---
seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)

st.title("Prediksi Gula Pereduksi Berdasarkan Nilai Brix dengan K-Fold")

# --- Data Real dan Interpolasi ---
nilai_brix = [73.0, np.nan, 76.0, np.nan, 75.5, np.nan, 73.6, np.nan, 76.0, np.nan,
              73.5, 77.5, 72.0, 76.8, 75.6, 78.5, 80.5, 79.5, 81.5, 83.0]
gula_pereduksi = [30.01, np.nan, np.nan, np.nan, np.nan, np.nan,
                  45.66, np.nan, 55.26, np.nan,
                  np.nan, 37.85, np.nan, np.nan, 61.16,
                  np.nan, np.nan, np.nan, np.nan, np.nan]

df = pd.DataFrame({
    'Nilai brix': nilai_brix,
    'Gula Pereduksi': gula_pereduksi
})

# Interpolasi nilai yang hilang
df['Nilai brix'] = df['Nilai brix'].interpolate()
df['Gula Pereduksi'] = df['Gula Pereduksi'].interpolate()

# --- Preprocessing ---
scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()
X = scaler_x.fit_transform(df[['Nilai brix']])
y = scaler_y.fit_transform(df[['Gula Pereduksi']])

# --- Input Epoch ---
epoch_input = st.number_input("Masukkan jumlah epoch", min_value=100, max_value=1000, value=100, step=100)

# --- Latih Model dengan K-Fold ---
if st.button("Latih Model dengan K-Fold"):
    with st.spinner("Melatih model terbaik..."):
        best_mse = float("inf")
        best_model = None
        history_best = None

        kf = KFold(n_splits=5, shuffle=True, random_state=seed)

        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            model = Sequential()
            model.add(Input(shape=(1,)))
            model.add(Dense(1, activation='linear'))  # JST Linear
            model.compile(optimizer=SGD(learning_rate=0.01), loss='mse')
            history = model.fit(X_train, y_train, epochs=int(epoch_input), verbose=0)

            y_pred = model.predict(X_test, verbose=0)
            mse = mean_squared_error(y_test, y_pred)

            if mse < best_mse:
                best_mse = mse
                best_model = model
                history_best = history

        st.session_state.model = best_model
        st.session_state.history = history_best
        st.session_state.scaler_x = scaler_x
        st.session_state.scaler_y = scaler_y
        st.success(f"✅ Model berhasil dilatih! MSE terbaik: {best_mse:.4f}")

# --- Setelah Pelatihan ---
if 'model' in st.session_state:
    model = st.session_state.model
    history = st.session_state.history
    scaler_x = st.session_state.scaler_x
    scaler_y = st.session_state.scaler_y

    # Bobot dan Bias
    weights, biases = model.layers[0].get_weights()
    st.write(f"**Bobot (Weight):** {weights[0][0]:.4f}")
    st.write(f"**Bias:** {biases[0]:.4f}")

    # Grafik Loss
    fig, ax = plt.subplots()
    ax.plot(history.history['loss'], label='Train Loss')
    ax.set_title('Loss Curve')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE')
    ax.legend()
    st.pyplot(fig)

    # Scatter Plot: Prediksi vs Aktual
    st.subheader("📊 Visualisasi: Prediksi vs Aktual")
    y_pred = model.predict(X, verbose=0)
    y_real = scaler_y.inverse_transform(y)
    y_pred_real = scaler_y.inverse_transform(y_pred)

    fig2, ax2 = plt.subplots()
    ax2.scatter(y_real, y_pred_real, color='blue', label='Prediksi')
    ax2.plot([y_real.min(), y_real.max()],
             [y_real.min(), y_real.max()],
             'r--', label='Garis Ideal (y = x)')
    ax2.set_xlabel("Aktual Gula Pereduksi")
    ax2.set_ylabel("Prediksi Gula Pereduksi")
    ax2.set_title("Scatter Plot: Prediksi vs Aktual")
    ax2.legend()
    st.pyplot(fig2)

    # Prediksi Manual
    brix_input = st.number_input("Masukkan nilai Brix", min_value=0.0, max_value=100.0, step=0.1)
    if st.button("Prediksi"):
        brix_df = pd.DataFrame({'Nilai brix': [brix_input]})
        brix_norm = scaler_x.transform(brix_df)
        pred_norm = model.predict(brix_norm, verbose=0)
        pred_real = scaler_y.inverse_transform(pred_norm)
        hasil = pred_real[0][0]

        st.success(f"Prediksi Gula Pereduksi: {hasil:.2f}%")
        if hasil >= 65:
            st.info("✅ Sesuai SNI (≥ 65%)")
        elif hasil >= 55:
            st.warning("⚠️ Sesuai untuk lebah tanpa sengat (≥ 55%)")
        else:
            st.error("❌ Tidak Sesuai SNI")
else:
    st.warning("🔁 Harap latih model terlebih dahulu dengan K-Fold.")

# Tampilkan Data Asli
with st.expander("📋 Lihat Data Asli (Interpolasi)"):
    st.dataframe(df)
