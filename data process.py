import json

import pandas as pd
from plotly.data import experiment


def select_median_value(data, output_path="bandgap_data_median.json"):
    # for multiple experimental band gaps that match same formula, select the median value of the band gaps
    # Convert the Value column from string to numeric list
    data['Value'] = data['Value'].str.strip('[]').astype(str).str.split(',').apply(lambda x: list(map(float, x)))

    # Expand rows with list values into separate rows
    data = data.explode('Value')
    data['Value'] = data['Value'].astype(float)

    # Group by the formula column and calculate the median for the Value column
    result = data.groupby('formula', as_index=False).agg({
        'mpids': 'first',
        'is_stable': 'first',
        'theoretical': 'first',
        'band_gap': 'first',
        'formation_energy_per_atom': 'first',
        'Value': 'median',
        'DOI': 'first',
        'compound_type': 'first'
    })

    # Save the result to a new CSV file
    result.to_csv('bandgap_data_median.csv', index=False)
    print("Processed data saved to 'bandgap_data_median.csv'")

    # Generate JSON file with mpids as keys and Value as bg
    data_json = result.set_index('mpids')['Value'].to_dict()
    data_json = {key: {"bg": value} for key, value in data_json.items()}

    # Save the JSON to a file
    with open(output_path, 'w') as json_file:
        json.dump(data_json, json_file, indent=4)

    print("Processed data saved to", output_path)

def split_data(data, split_ratio=0.8):
    # Split the data into training and validation sets
    train_data = data.sample(frac=split_ratio, random_state=42)
    val_data = data.drop(train_data.index)

    return train_data, val_data

# Function to remove overlapping materials from ds1 and save the result
def remove_overlapping(computational_ds, experimental_ds, output_path="new_ds1.json"):


        # Extract material IDs (keys) from both datasets
        materials_ds1 = set(computational_ds.keys())
        materials_ds2 = set(experimental_ds.keys())

        # Find overlapping material IDs
        overlap_ds1_ds2 = materials_ds1.intersection(materials_ds2)

        # Remove overlapping materials from ds1
        new_ds1 = {key: computational_ds[key] for key in materials_ds1 if key not in overlap_ds1_ds2}

        # Save the updated ds1 as a new JSON file
        with open(output_path, 'w') as json_file:
            json.dump(new_ds1, json_file, indent=4)

        # Print results
        print(f"Total materials in computational_ds: {len(materials_ds1)}")
        print(f"Total materials in experimental_ds: {len(materials_ds2)}")
        print(f"Number of overlapping materials: {len(overlap_ds1_ds2)}")
        print(f"Number of materials in new computational_ds: {len(new_ds1)}")

def csv_to_json(csv_path, json_path):
    # Read the CSV file
    data = pd.read_csv(csv_path)

    # Generate JSON file with mpids as keys and Value as bg
    data_json = data.set_index('mpids')['band_gap'].to_dict()
    data_json = {key: {"bg": value} for key, value in data_json.items()}

    # Save the JSON to a file
    with open(json_path, 'w') as json_file:
        json.dump(data_json, json_file, indent=4)

    print("Processed data saved to", json_path)

    print(f"Data saved to {json_path}")

data = pd.read_csv('final_data.csv')
select_median_value(data, output_path="bandgap_data_median.json")
# Paths to the JSON files

with open("pretrain_data.json", "r") as fp:
    computational_ds = json.load(fp)
with open("bandgap_data_median.json", "r") as fp:
    experimental_ds = json.load(fp)


# Call the function to compare and generate the new dataset
remove_overlapping(computational_ds, experimental_ds, output_path="pretrain_data.json")

# remove experimental data with bg < 0.5 or bg>5
experimental_ds_filtered = {key: value for key, value in experimental_ds.items() if 0.5 <= value['bg'] <= 5}
# Save the filtered experimental dataset to a new JSON file
with open("experimental_data_filtered.json", "w") as json_file:
    json.dump(experimental_ds_filtered, json_file, indent=4)
# Convert experimental_ds_filtered dictionary to a DataFrame
experimental_ds_filtered_df = pd.DataFrame.from_dict(experimental_ds_filtered, orient="index")
experimental_ds_filtered_df.reset_index(inplace=True)
experimental_ds_filtered_df.rename(columns={"index": "mpids"}, inplace=True)

# Now split the data using the DataFrame
train_data, test_data = split_data(experimental_ds_filtered_df, split_ratio=0.9)

# Save the train and test datasets to JSON files
train_data.set_index("mpids", inplace=True)
train_data_json = train_data.to_dict(orient="index")
test_data.set_index("mpids", inplace=True)
test_data_json = test_data.to_dict(orient="index")

# Save the converted datasets
with open("filtered_data/train_data.json", "w") as json_file:
    json.dump(train_data_json, json_file, indent=4)
with open("filtered_data/test_data.json", "w") as json_file:
    json.dump(test_data_json, json_file, indent=4)