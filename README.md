# Clothing Search Engine - Information Retrieval Assignment 1

## Project Overview
This project is a small-scale clothing search engine built using a corpus of 100 product descriptions. It features both a standard inverted index and a positional index to process free-text, exact phrase, and proximity queries.

## Team Members
* Sannidhya Rai
* Suhani Rohilla

## Features & Implementation
* **Vector Space Model (VSM):** Implements ranked retrieval for free-text queries using the `lnc.ltc` cosine similarity weighting scheme.
* **Positional Indexing:** Extends the inverted index to store token positions, enabling exact phrase searches and ordered proximity searches (WITHIN/k).
* **Pre-processing:** Includes tokenization, case normalization, punctuation removal, stop-word filtering, and stemming.

### Novelty Enhancements
To go beyond the baseline requirements, this project includes the following custom features:
1. **Custom Domain-Specific Stemmer:** A lightweight, rule-based stemmer tailored specifically for clothing and retail terminology, avoiding heavy third-party dependencies.
2. **Automated Evaluation Suite:** The script automatically executes all mandatory test queries (free-text, exact phrase, and proximity) upon startup to instantly validate the index.
3. **Path-Based Proximity Algorithm:** Utilizes an advanced position-tracking algorithm to accurately calculate valid sequence chains for multi-term proximity queries.
4. **Interactive CLI:** A robust, menu-driven command-line interface that allows the user to seamlessly switch between search modes.

## How to Run
1. Ensure Python 3.x is installed on your machine.
2. Verify that `corpus_100.txt` is in the same directory as the Python script.
3. Execute the script via terminal:
   ```bash
   python search_engine.py
