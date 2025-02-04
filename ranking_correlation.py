import itertools
import json
import pandas as pd
from scipy.stats import spearmanr
from itertools import combinations

# Load the JSON data from a file
with open('data/ds2.json', 'r') as file:
    data = json.load(file)

# Convert JSON data to a pandas DataFrame
df = pd.DataFrame.from_dict(data, orient='index')

# Rank the bg_mp and bg_exp values
df['rank_bg_mp'] = df['bg_mp'].rank()
df['rank_bg_exp'] = df['bg_exp'].rank()

# Calculate the Spearman correlation between the ranks
correlation, _ = spearmanr(df['rank_bg_mp'], df['rank_bg_exp'])

# Display the correlation value
print(f'Spearman correlation between bg_mp and bg_exp rankings: {correlation}')
# List to store comparison outcomes
correct_comparisons = 0
total_comparisons = 0

# Get all possible pairs of materials
pairs = list(itertools.combinations(df.index, 2))

# Iterate through each pair of materials
for mat1, mat2 in pairs:
    # Extract mp and exp values for the pair
    bg_mp1, bg_mp2 = df.loc[mat1, 'bg_mp'], df.loc[mat2, 'bg_mp']
    bg_exp1, bg_exp2 = df.loc[mat1, 'bg_exp'], df.loc[mat2, 'bg_exp']

    # Determine the comparison results
    mp_comparison = int(bg_mp1 > bg_mp2) - int(
        bg_mp1 < bg_mp2)  # 1 if bg_mp1 > bg_mp2, -1 if bg_mp1 < bg_mp2, 0 if equal
    exp_comparison = int(bg_exp1 > bg_exp2) - int(
        bg_exp1 < bg_exp2)  # 1 if bg_exp1 > bg_exp2, -1 if bg_exp1 < bg_exp2, 0 if equal

    # Check if the comparison matches
    if mp_comparison == exp_comparison:
        correct_comparisons += 1
    total_comparisons += 1

# Calculate accuracy
accuracy = correct_comparisons / total_comparisons if total_comparisons > 0 else 0

# Display the result
print(f'Accuracy of mp comparison against experimental comparison: {accuracy:.4f}')