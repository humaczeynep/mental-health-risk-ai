import streamlit as st
import pandas as pd
import joblib

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
        "En Çok Kullanılan Platform",
        ["Instagram", "TikTok", "X", "YouTube", "LinkedIn", "Snapchat"]
    )

    archetype = {
        "Aşırı Bağlı Kullanıcı": "Hyper-Connected",
        "Dengeli Kullanıcı": "Balanced User",
        "Pasif Kullanıcı": "Passive Scroller",
        "İçerik Üreticisi": "Content Creator"
    }[
        st.selectbox(
            "Kullanıcı Profili",
            ["Aşırı Bağlı Kullanıcı", "Dengeli Kullanıcı", "Pasif Kullanıcı", "İçerik Üreticisi"]
        )
    ]

with col2:
    screen_time = st.slider("Günlük Ekran Süresi (Saat)", 0.0, 12.0, 4.0, step=0.5)
    sleep_time = st.slider("Uyku Süresi (Saat)", 0.0, 12.0, 7.0, step=0.5)

    late_night = 1 if st.radio("Gece Kullanımı", ["Evet", "Hayır"]) == "Evet" else 0
    social_comparison = 1 if st.radio("Sosyal Karşılaştırma", ["Evet", "Hayır"]) == "Evet" else 0

activity_type = {"Aktif": "Active", "Pasif": "Passive"}[
    st.selectbox("Kullanım Şekli", ["Aktif", "Pasif"])
]

content_type = {
    "Yaşam Tarzı / Moda": "Lifestyle/Fashion",
    "Eğlence / Mizah": "Entertainment/Comedy",
    "Kişisel Gelişim / Motivasyon": "Self-Help/Motivation",
    "Eğitim / Teknoloji": "Educational/Tech",
    "Haberler / Politika": "News/Politics",
    "Oyun": "Gaming"
}[
    st.selectbox(
        "İçerik Türü",
        ["Yaşam Tarzı / Moda", "Eğlence / Mizah", "Kişisel Gelişim / Motivasyon",
         "Eğitim / Teknoloji", "Haberler / Politika", "Oyun"]
    )
]

st.markdown("---")


if st.button("Analiz Et", use_container_width=True):

    risk_score = (screen_time / 12) + (1 - sleep_time / 12)

    input_df = pd.DataFrame({
        "Age": [age],
        "Gender": [gender],
        "User_Archetype": [archetype],
        "Primary_Platform": [platform],
        "Daily_Screen_Time_Hours": [screen_time],
        "Dominant_Content_Type": [content_type],
        "Activity_Type": [activity_type],
        "Late_Night_Usage": [late_night],
        "Social_Comparison_Trigger": [social_comparison],
        "Sleep_Duration_Hours": [sleep_time],
        "Risk_Score": [risk_score]
    })


    prediction = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0]

    label_map = {0: "Düşük", 1: "Minimal", 2: "Orta", 3: "Yüksek"}

    if prediction == 0:
        st.success("🟢 Düşük Kaygı Riski")
    elif prediction == 1:
        st.info("🔵 Minimal Kaygı Riski")
    elif prediction == 2:
        st.warning("🟠 Orta Düzey Kaygı Riski")
    else:
        st.error("🔴 Yüksek Kaygı Riski")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Risk Skoru", f"{risk_score:.2f}")

    with col2:
        st.metric("Tahmin", label_map[prediction])


)

