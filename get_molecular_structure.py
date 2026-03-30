import pandas as pd
import requests
import os
import time
import datetime

# 2. Load the compounds from the previously extracted CSV
# (Make sure 'skiprows=3' matches the formatting of your specific CSV file)
csv_file_path = r"D:\SAF_Nurul\20260303_DrHA\compounds2.xlsx"
csv_sheet_name = 'bio_crude_sim'
df = pd.read_excel(csv_file_path, sheet_name=csv_sheet_name)
# Filter out 'Unknowns' and empty rows to get a clean list of unique compound names
compounds = [c for c in df['Name'].dropna().unique() if c != 'Unknowns']

# 1. Setup the output directory
output_dir =os.path.join(os.path.dirname(csv_file_path),  "molecule_images_" + datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
os.makedirs(output_dir, exist_ok=True)

print(f"Starting image downloads for {len(compounds)} compounds...")

# 3. Iterate through the list and download images
for i, compound in enumerate(compounds, 1):
    # The PubChem API endpoint for retrieving PNG images by chemical name
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{compound}/PNG"

    # Replace invalid characters (like slashes or commas) with underscores for safe file saving
    safe_filename = "".join([c if c.isalnum() else "_" for c in compound])
    filepath = os.path.join(output_dir, f"{safe_filename}.png")

    try:
        # Add verify=False to ignore the SSL error
        response = requests.get(url, verify=False)

        # Status code 200 means the image was successfully found and retrieved
        if response.status_code == 200:
            with open(filepath, 'wb') as f:
                f.write(response.content)
            print(f"[{i}/{len(compounds)}] Saved: {filepath}")
        else:
            print(f"[{i}/{len(compounds)}] Image not found for: {compound}")

    except Exception as e:
        print(f"[{i}/{len(compounds)}] Error downloading {compound}: {e}")

    except Exception as e:
        print(f"[{i}/{len(compounds)}] Error downloading {compound}: {e}")

    # PubChem's strict usage policy requires limiting requests to 5 per second.
    # A 0.25-second delay prevents your IP address from being blocked.
    time.sleep(0.25)

print("\nImage download process complete. Check the 'molecule_images' folder.")
