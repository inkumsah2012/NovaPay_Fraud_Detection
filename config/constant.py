import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

Input_Data = os.path.join(BASE_DIR, "Data", "nova_pay_combined.csv")
Cleaned_Data = os.path.join(
    BASE_DIR,
    "Data",
    "cleaned_data.csv"
)

Feature_Engineered_Data = os.path.join(
    BASE_DIR,
    "Data",
    "feature_engineered_data.csv"
)