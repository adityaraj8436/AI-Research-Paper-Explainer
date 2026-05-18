import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from model import train_and_predict

st.set_page_config(page_title="Glacier AI", layout="wide")

# Sidebar
st.sidebar.title("⚙️ Controls")
epochs = st.sidebar.slider("Epochs", 5, 50, 10)
look_back = st.sidebar.slider("Look Back", 3, 12, 5)

st.title("🌍 Glacier Mass Prediction using AI (LSTM)")
st.write("Upload dataset to predict future glacier mass trends.")

# Upload file
uploaded_file = st.file_uploader("📂 Upload CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Validation
    if 'Mass' not in df.columns:
        st.error("❌ CSV must contain 'Mass' column")
        st.stop()

    if len(df) < 30:
        st.error("❌ Dataset too small (need at least 30 rows)")
        st.stop()

    st.success("✅ File loaded")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Data Preview")
        st.dataframe(df.head())

    with col2:
        st.subheader("📈 Trend")
        fig, ax = plt.subplots()
        ax.plot(df['Mass'], label="Actual")
        ax.legend()
        st.pyplot(fig)

    if st.button("🚀 Train & Predict"):

        data = df[['Mass']].values

        with st.spinner("Training model..."):
            try:
                future_preds, train_rmse, test_rmse, test_mae = train_and_predict(
                    data, look_back, epochs
                )
            except Exception as e:
                st.error(str(e))
                st.stop()

        st.success("✅ Prediction Completed")

        # Metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Train RMSE", f"{train_rmse:.2f}")
        col2.metric("Test RMSE", f"{test_rmse:.2f}")
        col3.metric("Test MAE", f"{test_mae:.2f}")

        # Future predictions
        future_df = pd.DataFrame({
            "Month": range(1, 13),
            "Predicted Mass": future_preds.flatten()
        })

        st.subheader("🔮 Future Predictions")
        st.dataframe(future_df)

        # Plot predictions
        fig2, ax2 = plt.subplots()
        ax2.plot(df['Mass'], label="Actual")

        ax2.plot(
            range(len(df), len(df) + 12),
            future_preds.flatten(),
            linestyle='dashed',
            marker='o',
            label="Future"
        )

        ax2.legend()
        st.pyplot(fig2)

        # Download
        csv = future_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Predictions", csv, "predictions.csv")

else:
    st.warning("⚠️ Upload dataset to begin")