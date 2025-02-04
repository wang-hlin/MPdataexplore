import pandas as pd
import os
import json
import yaml

# Read the json file
data = pd.read_json('ablation.json')
data = data.T
# Add mpids as a new column
data['mpids'] = data.index

# Ensure 'type' exists in the data
if 'type' not in data.columns:
    raise ValueError("'type' column is missing in the input JSON file.")

# Create the output folder
output_folder = 'data_by_type'
os.makedirs(output_folder, exist_ok=True)

# Group by 'type' and generate JSON files
for compound_type, group in data.groupby('type'):
    # Create a dictionary for each compound type
    grouped_data = {row['mpids']: {"bg": row['bg']} for _, row in group.iterrows()}
    
    # Define the JSON file name
    file_name = f"bandgap_data_{compound_type}.json"
    output_path = os.path.join(output_folder, file_name)
    
    # Write to a JSON file
    with open(output_path, 'w') as json_file:
        json.dump(grouped_data, json_file, indent=4)

print(f"JSON files have been saved to the folder: {output_folder}")

# Loop over the unique categories in `compound_type`
for compound_type in data['type'].unique():
    # Create a folder for this category
    output_folder = f"data_by_type/data_{compound_type}"
    os.makedirs(output_folder, exist_ok=True)
    
    # Split data into test and train
    test_data = data[data['type'] == compound_type]
    train_data = data[data['type'] != compound_type]
    
    # Create dictionaries for test and train data
    test_data_dict = {row['mpids']: {"bg": row['bg']} for _, row in test_data.iterrows()}
    train_data_dict = {row['mpids']: {"bg": row['bg']} for _, row in train_data.iterrows()}
    
    # Write `test_data.json`
    with open(os.path.join(output_folder, 'test_data.json'), 'w') as test_file:
        json.dump(test_data_dict, test_file, indent=4)
    
    # Write `train_data.json`
    with open(os.path.join(output_folder, 'train_data.json'), 'w') as train_file:
        json.dump(train_data_dict, train_file, indent=4)

print("Datasets have been created for each category.")

# generate yaml files for each compound type to train finetuning models
# Base configuration template
base_config = {
    "DATASET": {
        "TRAIN": "data/experiment_bg/train_data.json",  # Placeholder
        "VAL": "data/experiment_bg/test_data.json"      # Placeholder
    },
    "MODEL": {
        "NAME": 'random_forest',
        "PRETRAINED_MODEL_PATH": '_saved_models/leftnet/m1_full_lr0.01_encoding/best-mae-epoch=04-val_mae=0.43.ckpt',
        "INIT_FILE": "master_files/encoding/atom_init.json",
        "CIF_FOLDER": "cif_file"
    },
    "SOLVER": {
        "LR": 0.001,
        "TRAIN_RATIO": 0.9,
        "VAL_RATIO": 0.1,
        "TEST_RATIO": None,
        "EPOCHS": 50,
        "NUM_RUNS": 5
    },
    # "LEFTNET": {
    #     "ENCODING": "one-hot",
    #     "LAYER_FREEZE": "none"
    # },
    "LOGGING": {
        "LOG_DIR": "saved_models",  # Keeps the main directory
        "LOG_DIR_NAME": "leftnet/ds2_leftnet_encoding_embedding"  # Placeholder, to be updated per category
    },
    "OUTPUT": {
        "DIR": "predictions/random_forest" # Placeholder
    }
}

# Categories for which to generate config files
categories = data['type'].unique()

# Output directory for config files
output_folder = "configs_by_category/filtered_data_random_forest"
os.makedirs(output_folder, exist_ok=True)

# Generate a YAML config file for each category
for category in categories:
    # Update dataset paths and log directory name for the current category
    category_config = base_config.copy()
    category_config["DATASET"]["TRAIN"] = f"data/ablation_filtered_data/data_{category}/train_data.json"
    category_config["DATASET"]["VAL"] = f"data/ablation_filtered_data/data_{category}/test_data.json"
    category_config["LOGGING"]["LOG_DIR_NAME"] = f"ablation_filtered_data/random_forest/{category}"
    category_config["OUTPUT"]["DIR"] = f"predictions/random_forest/{category}"
    
    # Generate file path
    config_file_path = os.path.join(output_folder, f"config_{category}.yaml")
    
    # Write YAML config
    with open(config_file_path, 'w') as yaml_file:
        yaml.dump(category_config, yaml_file, default_flow_style=False)
        
print(f"YAML config files have been created in the folder: {output_folder}")