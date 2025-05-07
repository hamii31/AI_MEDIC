#!/bin/bash

# Install Python dependencies
pip install -r requirements.txt

# Download the SpaCy model
python -m spacy download en_core_web_sm

# Exit with success
exit 0
