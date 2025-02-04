import os
import pandas as pd
from mp_api.client import MPRester

# MPRester API Key
API_KEY = "6TvioMOojWF4nSgvYrxm66br5f4YQCd0"

# Output directories and files
cif_directory = "cif_file"
failed_mpid_file = "failed_mpids.csv"
data_file = "selected_data.csv"

# Create the CIF directory if it doesn't exist
if not os.path.exists(cif_directory):
    os.makedirs(cif_directory)

# List to store failed mpids
failed_mpids = []

# Start retrieving data
with MPRester(API_KEY) as mpr:
    print("Retrieving materials data...")
    docs = mpr.materials.summary.search(
        num_elements=(0, 8),
        band_gap=(0.5, 5),
        fields=[
            "material_id",
            "formula_pretty",
            "elements",
            "num_elements",
            "is_stable",
            "theoretical",
            "band_gap",
            "formation_energy_per_atom",
        ],
    )

    # Create a list of material data
    data = []
    for doc in docs:
        data.append({
            "mpids": doc.material_id,
            "formula": doc.formula_pretty,
            "elements": doc.elements,
            "is_stable": doc.is_stable,
            "theoretical": doc.theoretical,
            "band_gap": doc.band_gap,
            "formation_energy_per_atom": doc.formation_energy_per_atom,
        })

    # Process each material and save CIF files
    print("Processing materials and saving CIF files...")
    for material in data:
        mpid = material["mpids"]
        try:
            # Retrieve structure and save as CIF
            structure = mpr.get_structure_by_material_id(mpid)
            cif_filename = os.path.join(cif_directory, f"{mpid}.cif")
            structure.to(fmt="cif", filename=cif_filename)
            print(f"Saved CIF for {mpid}")
        except Exception as e:
            print(f"Failed to process {mpid}: {e}")
            failed_mpids.append(mpid)  # Store the failed mpid

# Save the retrieved data to CSV
df = pd.DataFrame(data)
df.to_csv(data_file, index=False)
print(f"Saved retrieved data to {data_file}")

# Save failed mpids to a CSV file
if failed_mpids:
    failed_df = pd.DataFrame({"failed_mpids": failed_mpids})
    failed_df.to_csv(failed_mpid_file, index=False)
    print(f"Saved failed mpids to {failed_mpid_file}")
else:
    print("All mpids were successfully processed!")
