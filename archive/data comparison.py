import ast
import re

import numpy as np
import pandas as pd
from fontTools.subset import subset

###
# filter the data from Chemdataextractor, compare it with data from Materials Project

# def categorize_compound(formula):
#     # Define regular expressions for each category
#     categories = {
#         "Oxides": r"O(?![A-Za-z])",
#         "Chalcogenides": r"S|Se|Te",
#         "Nitrides": r"N(?![A-Za-z])",
#         "Phosphides": r"P(?![A-Za-z])",
#         "Arsenides": r"As",
#         "Halides": r"F|Cl|Br|I|At",
#         "Antimonides": r"Sb",
#         "Silicides": r"Si",
#         "Carbides": r"C(?![A-Za-z])",
#         "Hydrides": r"H(?![A-Za-z])",
#         "Double anions": r"[A-Za-z]+\d?2",
#         "Others": None  # Default category
#     }
#
#     # Check each category in order
#     for category, pattern in categories.items():
#         if pattern and re.search(pattern, formula):
#             return category
#
#     # If no category matched, return "Others"
#     return "Others"
#%%

def categorize_compound(formula):
    # Define regular expressions for each category, ordered by priority
    categories = {
        "Double anions": r"[A-Za-z]+\d?2",
        "Halides": r"F|Cl|Br|I",
        "Chalcogenides": r"S|Se|Te",
        "Oxides": r"O(?![A-Za-z])",
        "Nitrides": r"N(?![A-Za-z])",
        "Phosphides": r"P(?![A-Za-z])",
        "Arsenides": r"As",
        "Antimonides": r"Sb",
        "Silicides": r"Si",
        "Carbides": r"C(?![A-Za-z])",
        "Hydrides": r"H(?![A-Za-z])",
        "Others": None  # Default category
    }

    matched_categories = []

    # Check each category in priority order
    for category, pattern in categories.items():
        if pattern and re.search(pattern, formula):
            matched_categories.append(category)

    # If multiple categories match, prioritize based on the first match
    if matched_categories:
        return matched_categories[0]  # Return the highest-priority match

    # If no category matched, return "Others"
    return "Others"
#%%
# Load the data

bg_df = pd.read_csv("Bandgap.csv")
print(bg_df)
bg_df = bg_df.dropna(subset=["DOI"])
print(bg_df)
mp_df = pd.read_csv("selected_data.csv")
# print distinct formula values in mp_df
print(mp_df["formula"].nunique())
# for the 'formula' column with same entry, keep the one with lowest 'formulation_energy_per_atom' value
mp_df = mp_df.sort_values("formation_energy_per_atom").drop_duplicates("formula", keep="first")
# print distinct formula values in mp_df
print(mp_df)

# join the two dataframes on the 'formula' column in mp_df and 'Name' column in bg_df
df = mp_df.merge(bg_df, left_on="formula", right_on="Name")
df = df.dropna(subset=["formula", "Value", "band_gap", "DOI"])

# only keep rows needed
df = df[["mpids", "formula", "is_stable", "theoretical", "band_gap", "formation_energy_per_atom", "Value","Temperature_raw_value", "DOI"]]
#
df['Value'] = df['Value'].apply(lambda x:[float(i) for i in ast.literal_eval(x)])
df_filtered = df[df['Value'].apply(len)==1]

print(df)
# print columns in df
print(df.columns)


# Apply the categorize_compound function to the 'formula' column and return the type of compound
df['compound_type'] = df['formula'].apply(categorize_compound)
df_filtered['compound_type'] = df_filtered['formula'].apply(categorize_compound)


print(f"Rows after filtering: {len(df_filtered)}")

df_filtered.to_csv("final_data.csv", index=False)

