import pandas as pd
import datetime as dt

def preprocessDate(df):
    
    df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d", errors="coerce")

    # Check for any conversion issues
    if df["Date"].isna().any():
        print("Warning: Some dates could not be converted. Check for incorrect formats.")

    # Convert Date format to MM/DD/YYYY
    df["Date"] = df["Date"].dt.strftime("%d/%m/%Y")

    return df
