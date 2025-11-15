#!/bin/bash

# This script downloads the GTZAN dataset from a reliable mirror on Kaggle
# and extracts it into the correct folder structure.

# URL for the GTZAN dataset (Kaggle mirror)
DATASET_URL="https://storage.googleapis.com/kaggle-data-sets/3983/6869/bundle/archive.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20240401%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20240401T150935Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=19c9e8a759e5d4d38e2172f3e8b15d2a93883a910825310860d5c0b11867c4e5113d0920d315f6795f57a3d132b4999d3e86c0f209f924f71d5301826649f48a939f323c2a1383713a078e8152c93806f40b790d963366c3a8d6727285c54f59e661642c34a11f93f1013cb5352e693166a1a8a25c3ac3d463d12d22b512e9b86377e6833c82537f09315354e7d727b3b421a24d559c5d14486968987b76214228af2456e30019e072f0578b97d1b32d201c1061c0f0d2354c0e0e090a9d82136d1ed5180f684a0b27e8523c14c538a0f983638065b832b461f615371b20f09b30b42f1f51042790936e522f790c8855e968031e42a3a0e368729"
FILE_NAME="gtzan_dataset.zip"
EXTRACT_DIR="genres_original"

echo "Starting download of the GTZAN dataset..."

# Use curl to download the file
# -L handles redirects
# -o specifies the output filename
curl -L -o "$FILE_NAME" "$DATASET_URL"

# Check if download was successful
if [ $? -eq 0 ]; then
    echo "Download complete! Now extracting the files..."
    
    # Use unzip to extract the archive.
    # -o overwrites files without prompting
    # We expect this to contain the 'genres_original' directory
    unzip -o "$FILE_NAME"

    # Check if extraction was successful and the directory exists
    if [ -d "$EXTRACT_DIR" ]; then
        echo "Extraction complete. The audio files are now in the '$EXTRACT_DIR' directory."
        echo "You can now delete the zip file to save space:"
        echo "rm $FILE_NAME"
    else
        echo "Extraction failed or the expected '$EXTRACT_DIR' directory was not found."
    fi
else
    echo "An error occurred during the download."
    echo "Please check your internet connection."
fi

