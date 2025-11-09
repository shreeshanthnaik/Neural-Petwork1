import roboflow
import os

# --- PASTE YOUR KEY HERE ---
# Find this in your Roboflow Settings -> Workspace -> Roboflow API
YOUR_API_KEY = "Sc57fKcKbg05NcHATXJG"

if YOUR_API_KEY == "PASTE_YOUR_PRIVATE_API_KEY_HERE":
    print("="*50)
    print("ERROR: Please open the 'download_data.py' file")
    print("and paste your Roboflow Private API Key into")
    print("the YOUR_API_KEY variable.")
    print("="*50)
else:
    try:
        # 1. Authenticate directly with the API key
        rf = roboflow.Roboflow(api_key=YOUR_API_KEY)
        
        # 2. Get the project from the "public" workspace
        #    (We are using a public dataset, so this should work)
        project = rf.workspace("public").project("oxford-pets")
        
        # 3. Download the "by-species" version (cat/dog)
        print("Authenticating...")
        print("Downloading dataset (this may take a minute)...")
        dataset = project.version(2).download("yolov8")
        
        print(f"Dataset downloaded to: {dataset.location}")
        print("Download complete.")
        
    except Exception as e:
        print(f"An error occurred: {e}")
        print("\nPlease double-check that your API key is correct and try again.")
        print("Make sure you copied the 'Private API Key'.")