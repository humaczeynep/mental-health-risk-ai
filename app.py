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


if SHAP_AVAILABLE:

    input_transformed = preprocessor.transform(input_df)
    explainer = shap.TreeExplainer(xgb)
    shap_values = explainer.shap_values(input_transformed)

    feature_names = preprocessor.get_feature_names_out()

    if isinstance(shap_values, list):
        shap_values = shap_values[prediction]

    shap_values = shap_values[0].reshape(-1)

    shap_df = pd.DataFrame({
        "feature": feature_names[:len(shap_values)],
        "impact": shap_values[:len(feature_names)]
    })

    shap_df["abs"] = shap_df["impact"].abs()

    top_features = shap_df.sort_values("abs", ascending=False).head(5)


    top_features["feature"] = (
        top_features["feature"]
        .str.replace("cat__", "")
        .str.replace("remainder__", "")
    )


    feature_translation = {
        "Age": "Yaş",
        "Daily_Screen_Time_Hours": "Ekran Süresi",
        "Sleep_Duration_Hours": "Uyku Süresi",
        "Risk_Score": "Risk Skoru",
        "Late_Night_Usage": "Gece Kullanımı",
        "Social_Comparison_Trigger": "Sosyal Karşılaştırma",
        "Activity_Type_Active": "Aktif Kullanım",
        "Activity_Type_Passive": "Pasif Kullanım",
        "Primary_Platform_Facebook": "Facebook",
        "Primary_Platform_Instagram": "Instagram",
        "Primary_Platform_TikTok": "TikTok",
        "Primary_Platform_X": "X (Twitter)",
        "Primary_Platform_YouTube": "YouTube",
        "User_Archetype_Hyper-Connected": "Aşırı Bağlı Kullanıcı",
        "User_Archetype_Balanced User": "Dengeli Kullanıcı",
        "User_Archetype_Passive Scroller": "Pasif Kullanıcı",
        "User_Archetype_Content Creator": "İçerik Üreticisi",
        "User_Archetype_Average User": "Ortalama Kullanıcı",
        "User_Archetype_Digital Minimalist": "Dijital Minimalist",
        "Dominant_Content_Type_Self-Help/Motivation": "Kişisel Gelişim / Motivasyon",
        "Dominant_Content_Type_Educational/Tech": "Eğitim / Teknoloji",
        "Dominant_Content_Type_News/Politics": "Haberler / Politika",
        "Dominant_Content_Type_Lifestyle/Fashion": "Yaşam Tarzı / Moda",
        "Dominant_Content_Type_Entertainment/Comedy": "Eğlence / Mizah",
        "Dominant_Content_Type_Gaming": "Oyun"
    }

    top_features["feature"] = top_features["feature"].apply(
        lambda x: feature_translation.get(x, x)
    )

    st.subheader("🔍 Bu Sonuç Neden Oluştu?")

    for _, row in top_features.iterrows():
        st.write(f"• **{row['feature']}** → `{row['impact']:.3f}`")

    st.bar_chart(top_features.set_index("feature")["abs"])

    st.subheader("💡 Kişiselleştirilmiş Öneriler")

    for _, row in top_features.iterrows():
        f = row["feature"]

        if "Uyku" in f:
            st.info("💤 Uyku düzenini iyileştir.")

        elif "Ekran" in f:
            st.info("📱 Ekran süreni azalt.")

        elif "Gece" in f:
            st.info("🌙 Gece kullanımını sınırlandır.")

        elif "Sosyal" in f:
            st.info("🧠 Sosyal karşılaştırmayı azalt.")

else:
    st.warning("🔍 SHAP analizi şu anda cloud ortamında çalışmıyor")
