#!/bin/bash

ROOT_DIR="Data/cropped"

find "$ROOT_DIR" -type f -name "*.zip" | while read zipfile; do
    echo "Extracting $zipfile"
    
    # Extract into same directory as the zip
    unzip -q "$zipfile" -d "$(dirname "$zipfile")"
done

echo "All files extracted."