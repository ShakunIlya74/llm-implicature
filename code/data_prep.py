import pandas as pd
from pathlib import Path

def to_df(folder="data"):
    """
    input: the folder path that holds your data, default "data"
    output: pandas dataframe
    TODO: i will add separate column "utterance" that's a cleaned up version of 
    input without formatting (basically just the scenario)
    """
    root = Path(folder)
    dfs = []

    for file in root.rglob("*.jsonl"):
        rel_parts = file.relative_to(root).parts
        model = rel_parts[-4]
        method = rel_parts[-3]

        df = pd.read_json(file, lines=True)
        df["strat"] = method
        df["method"] = model

        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)

if __name__ == "__main__":
    folder = str(input("Enter folder: "))
    df = to_df(folder)
    csv = str(input("Dataframe created. Write to csv? (y/n)"))
    if csv.lower() ==  "y" or csv.lower() == "yes":
        where = str(input("Enter file name: "))
        df.to_csv(f"{where}.csv")
        print("File created.")