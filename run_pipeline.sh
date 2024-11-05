#!/bin/bash

# Run jobs_scraping script
echo "Running jobs_scraping/main.py..."
python jobs_scraping/main.py

# Check if jobs_scraping script was successful
if [ $? -ne 0 ]; then
    echo "Error: jobs_scraping/main.py failed."
    exit 1
fi

# Run jobs_embedding script
echo "Running jobs_embedding/main.py..."
python jobs_embedding/main.py

# Check if jobs_embedding script was successful
if [ $? -ne 0 ]; then
    echo "Error: jobs_embedding/main.py failed."
    exit 1
fi

# Run main script
echo "Running main.py..."
python main.py

# Check if main script was successful
if [ $? -ne 0 ]; then
    echo "Error: main.py failed."
    exit 1
fi

echo "All scripts ran successfully."
