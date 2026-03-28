# Text Embedding Model

Text classification model using word embeddings and LSTM networks for processing variable-length sequences.

## Architecture
- **Embedding Layer**: Converts word indices to dense vectors
- **LSTM Layer**: Bidirectional LSTM for sequence processing  
- **Classifier**: Fully connected layer for classification
- **Vocabulary**: Custom vocabulary with special tokens

## Text Processing
- Tokenization and vocabulary mapping
- Variable-length sequence handling
- Padding and sequence packing for efficiency
- Unknown token handling
- Text preprocessing pipeline

## Model Specifications
- Vocabulary size: Configurable (default ~10K words)
- Embedding dimension: 128
- Hidden dimension: 256 (bidirectional = 512 total)
- Output classes: 5 (sentiment classification)

## Expected Functionality
1. Build vocabulary from text data
2. Tokenize and convert text to indices
3. Handle variable-length sequences with padding
4. Process through embedding and LSTM layers
5. Classify text into sentiment categories
6. Train with proper sequence length handling

## Usage
```bash
python embedding_model.py
```