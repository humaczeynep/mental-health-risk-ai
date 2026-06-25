import pandas as pd

df = pd.read_csv("data/social_media_mental_health.csv")

df = df.drop(columns=[
    "User_ID",
    "GAD_7_Score",
    "PHQ_9_Score",
    "PHQ_9_Severity"
])

X = df.drop("GAD_7_Severity", axis=1)
y = df["GAD_7_Severity"]