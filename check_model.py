# check_model.py
import joblib
import pandas as pd
import traceback

MODEL_PATH = "models/dynamic_pricing_model.pkl"

def main():
    try:
        m = joblib.load(MODEL_PATH)
        print("Model loaded:", type(m))
        if hasattr(m, "feature_names_in_"):
            print("feature_names_in_:", list(m.feature_names_in_))
        # try a dummy predict if model supports it
        try:
            # build a dummy row with some common fields from your UI
            row = {
                "Item_Identifier": "FDW12",
                "Item_Weight": 10.0,
                "Item_Fat_Content": "Regular",
                "Item_Visibility": 0.035,
                "Item_Type": "Baking Goods",
                "Item_MRP": 144.54,
                "Outlet_Establishment_Year": 1985,
                "Outlet_Size": "Medium",
                "Outlet_Location_Type": "Tier 3",
                "Outlet_Type": "Supermarket Type3",
                "Outlet_Age": 40,
                "avg_sales_by_item": 0.0,
                "avg_sales_by_outlet_item": 0.0,
                "item_category_aggregates": 0.0,
                "is_perishable": 0
            }
            df = pd.DataFrame([row])
            print("Trying predict(...)")
            preds = m.predict(df)
            print("Prediction result:", preds)
        except Exception as e:
            print("Predict failed (maybe model needs different preprocessing):", e)
            traceback.print_exc()
    except Exception as e:
        print("Failed to load model:", e)
        traceback.print_exc()

if __name__ == "__main__":
    main()
