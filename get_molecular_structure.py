import pandas as pd
import requests
import os
import time
import datetime
import urllib3

# Suppress the insecure request warnings caused by verify=False
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# 2. Load the compounds from the previously extracted CSV
csv_file_path = r"D:\SAF_Nurul\20260303_DrHA\compounds2.xlsx"
csv_sheet_name = 'bio_crude_sim'
df = pd.read_excel(csv_file_path, sheet_name=csv_sheet_name)

# Filter out 'Unknowns' and empty rows, keeping the whole row to preserve 'Molecular ID'
df = df.dropna(subset=['Name'])
df = df[df['Name'] != 'Unknowns']
# Drop duplicate names so we only download one image per unique compound
unique_compounds_df = df.drop_duplicates(subset=['Name'])

# 1. Setup the output directory
output_dir = os.path.join(os.path.dirname(csv_file_path),
                          "molecule_images_" + datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
os.makedirs(output_dir, exist_ok=True)

print(f"Starting image downloads for {len(unique_compounds_df)} compounds...")

# 3. Iterate through the DataFrame rows to access both Name and Molecular ID
for i, row in enumerate(unique_compounds_df.to_dict('records'), 1):
    compound = str(row['Name'])
    mol_id = row.get('Molecular ID')  # Use .get() in case the column is missing

    # Replace invalid characters (like slashes or commas) with underscores for safe file saving
    safe_filename = "".join([c if c.isalnum() else "_" for c in compound])
    filepath = os.path.join(output_dir, f"{safe_filename}.png")

    # The PubChem API endpoint for retrieving PNG images by chemical name
    url_name = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{compound}/PNG"

    try:
        # ATTEMPT 1: Search by Name
        response = requests.get(url_name, verify=False)

        if response.status_code == 200:
            with open(filepath, 'wb') as f:
                f.write(response.content)
            print(f"[{i}/{len(unique_compounds_df)}] Saved by Name: {compound}")

        else:
            # ATTEMPT 2: Fallback to Molecular ID if Name fails
            if pd.notna(mol_id):
                # Ensure the ID is formatted as a clean string (e.g., converting 123.0 to '123')
                clean_id = str(int(mol_id)) if isinstance(mol_id, (float, int)) else str(mol_id).strip()
                url_cid = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{clean_id}/PNG"

                response_cid = requests.get(url_cid, verify=False)

                if response_cid.status_code == 200:
                    with open(filepath, 'wb') as f:
                        f.write(response_cid.content)
                    print(f"[{i}/{len(unique_compounds_df)}] Saved by Molecular ID ({clean_id}): {compound}")
                else:
                    print(f"[{i}/{len(unique_compounds_df)}] Not found by Name or ID: {compound}")
            else:
                # Fails if Name doesn't work and there is no ID in the column
                print(f"[{i}/{len(unique_compounds_df)}] Not found by Name (No Molecular ID provided): {compound}")

    except Exception as e:
        print(f"[{i}/{len(unique_compounds_df)}] Error downloading {compound}: {e}")

    # PubChem's strict usage policy requires limiting requests to 5 per second.
    # A 0.25-second delay prevents your IP address from being blocked.
    time.sleep(0.25)

print(f"\nImage download process complete. Check the '{os.path.basename(output_dir)}' folder.")