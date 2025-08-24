# portfolio_json.py
import pandas as pd
import json

def dataframe_to_portfolio_json(df, output_file="Clients_Portfolios.json"):
    # If not overwritten, saves output JSON into Clients_Portfolios.json
    """
    Convert cleaned portfolio DataFrame into JSON snapshots for each client.
    (Only uses columns that exist in the CSV, no artificial fields.)
    """
    # Each row becomes a dictionary, keys = column names
    clients = df.to_dict(orient="records") #Converts the DataFrame into a list of dictionaries.
    #orient="records" means: each row becomes a dictionary, where keys = column names and values = row values.

    # Save to file
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(clients, f, indent=2, ensure_ascii=False)

    print(f"✅ Saved {len(clients)} client portfolios to {output_file}")
    return clients


if __name__ == "__main__":
    # Example usage: load cleaned CSV
    df = pd.read_csv("Cleaned_Active_Portfolio.csv")
    portfolios = dataframe_to_portfolio_json(df)
