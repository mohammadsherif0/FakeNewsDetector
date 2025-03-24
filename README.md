# Fake News Detection Project

This repository contains code for loading and processing various fake news detection datasets. The project is designed to work with multiple well-known fake news datasets while keeping the data loading process streamlined and reproducible.

## Project Structure

```
fake-news-detector/
├── data_loaders/          # Dataset loading utilities
├── model/                 # Model implementation and training
├── preprocessing/         # Text preprocessing pipeline
└── tests/                # Test scripts for each component
```

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

## Recent Improvements

1. **Dataset Management**
   - Implemented unified dataset class for all sources
   - Added support for test mode with smaller subsets
   - Improved error handling and validation
   - Added progress tracking for data loading

2. **Preprocessing Enhancements**
   - Standardized text cleaning across datasets
   - Improved label encoding with better error handling
   - Added data validation checks
   - Implemented efficient batch processing

3. **Testing Infrastructure**
   - Added comprehensive test suite
   - Implemented dataset-specific test scripts
   - Added validation for data consistency
   - Improved error reporting

## Upcoming Features

1. **Model Implementation**
   - BERT-based classifier for fake news detection
   - Integration with Hugging Face Trainer API
   - Support for fine-tuning and transfer learning
   - Model evaluation and metrics

2. **Training Pipeline**
   - Configurable training parameters
   - Model checkpointing and early stopping
   - Performance monitoring and logging
   - Cross-dataset evaluation

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

