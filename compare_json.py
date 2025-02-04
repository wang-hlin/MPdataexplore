import json

def compare_three_materials(ds1_path, ds2_path, ds3_path):
    # Load the JSON files
    with open(ds1_path, 'r') as ds1_file, open(ds2_path, 'r') as ds2_file, open(ds3_path, 'r') as ds3_file:
        ds1 = json.load(ds1_file)
        ds2 = json.load(ds2_file)
        ds3 = json.load(ds3_file)

    # Extract material IDs from all three files
    materials_ds1 = set(ds1.keys())
    materials_ds2 = set(ds2.keys())
    materials_ds3 = set(ds3.keys())

    # Find overlaps between ds1 and ds2
    overlap_ds1_ds2 = materials_ds1.intersection(materials_ds2)

    # Find if materials in ds3 are in ds1 or ds2
    overlap_ds3_ds1 = materials_ds3.intersection(materials_ds1)
    overlap_ds3_ds2 = materials_ds3.intersection(materials_ds2)
    overlap_ds3_ds1_or_ds2 = overlap_ds3_ds1.union(overlap_ds3_ds2)

    # Print results
    print(f"Total materials in ds1: {len(materials_ds1)}")
    print(f"Total materials in ds2: {len(materials_ds2)}")
    print(f"Total materials in ds3: {len(materials_ds3)}")
    print(f"Number of overlapping materials between ds1 and ds2: {len(overlap_ds1_ds2)}")
    print(f"Number of materials in ds3 that are in ds1: {len(overlap_ds3_ds1)}")
    print(f"Number of materials in ds3 that are in ds2: {len(overlap_ds3_ds2)}")
    print(f"Number of materials in ds3 that are in either ds1 or ds2: {len(overlap_ds3_ds1_or_ds2)}")


ds1_path = "data/ds1.json"
ds2_path = "data/ds2.json"
ds3_path = "data/ds3.json"

compare_three_materials(ds1_path, ds2_path, ds3_path)
