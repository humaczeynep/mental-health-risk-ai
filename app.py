import streamlit as st
import pandas as pd
import joblib


try:
    import shap
    SHAP_AVAILABLE = True
except:
    SHAP_AVAILABLE = False


st.set_page_config(
    page_title="MindLens AI",
    page_icon="🧠",
    layout="centered"
)


model = joblib.load("mental_health_model.pkl")
preprocessor = model.named_steps["preprocessor"]
xgb = model.named_steps["model"]


st.title("🧠 MindLens AI")
st.caption("Yapay Zekâ Destekli Sosyal Medya ve Kaygı Risk Analizi")
st.markdown("---")


col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Yaş", 18, 22, 19)

    gender = {"Erkek": "Male", "Kadın": "Female"}[
        st.selectbox("Cinsiyet", ["Erkek", "Kadın"])
    ]

    platform = st.selectbox(
        "Platform",
        ["Instagram", "TikTok", "X", "YouTube", "LinkedIn", "Snapchat"]
    )

    archetype = {
        "Aşırı Bağlı": "Hyper-Connected",
        "Dengeli": "Balanced User",
        "Pasif": "Passive Scroller",
        "Üretici": "Content Creator"
    }[
        st.selectbox("Profil", ["Aşırı Bağlı", "Dengeli", "Pasif", "Üretici"])
    ]

with col2:
    screen_time = st.slider("Ekran Süresi", 0.0, 12.0, 4.0)
    sleep_time = st.slider("Uyku", 0.0, 12.0, 7.0)

    late_night = 1 if st.radio("Gece Kullanımı", ["Evet", "Hayır"]) == "Evet" else 0
    social = 1 if st.radio("Sosyal Karşılaştırma", ["Evet", "Hayır"]) == "Evet" else 0

activity = {"Aktif": "Active", "Pasif": "Passive"}[
    st.selectbox("Aktivite", ["Aktif", "Pasif"])
]

content = {
    "Lifestyle": "Lifestyle/Fashion",
    "Entertainment": "Entertainment/Comedy",
    "Self Help": "Self-Help/Motivation",
    "Education": "Educational/Tech",
    "News": "News/Politics",
    "Gaming": "Gaming"
}[
    st.selectbox("İçerik", ["Lifestyle", "Entertainment", "Self Help", "Education", "News", "Gaming"])
]

st.markdown("---")


if st.button("Analiz Et"):

    risk_score = (screen_time / 12) + (1 - sleep_time / 12)

    input_df = pd.DataFrame({
        "Age": [age],
        "Gender": [gender],
        "User_Archetype": [archetype],
        "Primary_Platform": [platform],
        "Daily_Screen_Time_Hours": [screen_time],
        "Dominant_Content_Type": [content],
        "Activity_Type": [activity],
        "Late_Night_Usage": [late_night],
        "Social_Comparison_Trigger": [social],
        "Sleep_Duration_Hours": [sleep_time],
        "Risk_Score": [risk_score]
    })

    pred = model.predict(input_df)[0]

    labels = {0: "Düşük", 1: "Minimal", 2: "Orta", 3: "Yüksek"}

    st.success(f"Sonuç: {labels[pred]}")

    st.metric("Risk Skoru", f"{risk_score:.2f}")

  
    if SHAP_AVAILABLE:
        try:
            X = preprocessor.transform(input_df)
            explainer = shap.TreeExplainer(xgb)
            shap_values = explainer.shap_values(X)

            feature_names = preprocessor.get_feature_names_out()

            if isinstance(shap_values, list):
                shap_values = shap_values[pred]

            shap_values = shap_values[0]

            shap_df = pd.DataFrame({
                "feature": feature_names[:len(shap_values)],
                "impact": shap_values[:len(feature_names)]
            })

            shap_df["abs"] = shap_df["impact"].abs()
            top = shap_df.sort_values("abs", ascending=False).head(5)

            st.subheader("🔍 En Etkili Faktörler")

            for _, r in top.iterrows():
                st.write(f"• {r['feature']} → {r['impact']:.3f}")

        except:
            st.warning("SHAP şu an gösterilemiyor")

    else:
        st.warning("SHAP cloud ortamında kapalı")
