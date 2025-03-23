# Fake News Detection Project

This repository contains code for loading and processing various fake news detection datasets. The project is designed to work with multiple well-known fake news datasets while keeping the data loading process streamlined and reproducible.

## Datasets

This project works with the following datasets:

1. **LIAR Dataset**
   - A benchmark dataset for fake news detection
   - Contains 12.8K human-labeled short statements
   - [Download here](https://www.cs.ucsb.edu/~william/data/liar_dataset.zip)

2. **FakeNewsNet**
   - Comprehensive dataset with news content and social context
   - Contains data from PolitiFact and GossipCop
   - [Available here](https://github.com/KaiDMML/FakeNewsNet)

3. **Fake and Real News Dataset**
   - Collection of real and fake news articles
   - [Available on Kaggle](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)

4. **Fake News Prediction Dataset**
   - Additional dataset for fake news classification
   - [Available on Kaggle](https://www.kaggle.com/datasets/rajatkumar30/fake-news)

## Preprocessing Pipeline

The project includes a comprehensive preprocessing pipeline that standardizes all datasets for model training:

1. **Text Cleaning**
   - Removes HTML tags, URLs, and special characters
   - Normalizes whitespace and text formatting
   - Preserves essential text content

2. **Tokenization**
   - Uses Hugging Face tokenizers compatible with transformer models
   - Handles text truncation and padding
   - Maintains consistent sequence lengths

3. **Label Encoding**
   - Standardizes labels across all datasets (0 for fake, 1 for real)
   - Handles multi-class labels where applicable
   - Preserves original label information in metadata

4. **Data Splitting**
   - Splits data into training, validation, and test sets
   - Maintains class balance across splits
   - Preserves dataset-specific splits where available

The preprocessing pipeline is implemented in the `preprocessing` directory and can be tested using the provided test scripts for each dataset.

## Why Datasets are Not Included

The datasets are not included in this repository for several reasons:
1. Large file sizes that would make the repository unwieldy
2. Respect for data providers' terms of service and distribution policies
3. Ensure users always get the most up-to-date version of the datasets
4. Avoid potential copyright and privacy issues

## Acknowledgments

- LIAR dataset creators
- FakeNewsNet team
- Kaggle community
- All other dataset providers

