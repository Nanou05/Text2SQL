# Text2SQL
This project implements two different approaches to converting natural language questions into SQL queries using transformer-based models:
BERT and Mistral-7B
The system is trained on the WikiSQL dataset and can generate SQL queries from natural language questions about database tables.

## Project Overview

The project explores and compares two different approaches:
1. Fine-tuning BERT for seq2seq conversion
2. Using Mistral-7B with LoRA for instruction-tuning

## Requirements

```bash
# Core dependencies
torch>=2.0.0
transformers>=4.30.0
datasets>=2.12.0
peft>=0.4.0
accelerate>=0.20.0
bitsandbytes>=0.40.0
trl>=0.7.2
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Text2sSQL.git
cd Text2sSQL
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
 Text2sSQL/
├── data/
│   └── wikisql/
├── models/
│   ├── bert_model/
│   └── mistral_model/
│   └── bard_model/
├── notebooks/
│   ├── data_analysis.ipynb
│   └── model_evaluation.ipynb
├── src/
│   ├── data_preprocessing.py
│   ├── bert_model.py
│   └── mistral_model.py
│   └── bard_model.py
├── README.md
└── requirements.txt
```

## BERT Approach

The BERT-based approach uses a seq2seq architecture with the following features:
- BERT encoder for natural language understanding
- Linear decoder for SQL query generation
- Cross-entropy loss for token-level prediction

### Training BERT Model

```python
from src.bert_model import TextToSQLBERT, train

# Initialize model
model = TextToSQLBERT()

# Train model
train(model, train_loader, val_loader, num_epochs=3)
```

## Mistral Approach

The Mistral-7B approach uses instruction tuning with LoRA for efficient fine-tuning:
- 4-bit quantization for memory efficiency
- LoRA adaptation for parameter-efficient fine-tuning
- Instruction-based prompting for better SQL generation

### Training Mistral Model

```python
from src.mistral_model import setup_model, train_model

# Setup model with LoRA
model, tokenizer = setup_model()

# Train model
train_model(model, tokenizer, train_dataset, val_dataset)
```
## Bard Approach

## Model Comparison

| Model | Exact Match Accuracy | Execution Accuracy | Training Time | Model Size |
|-------|---------------------|-------------------|---------------|------------|
| BERT  | %                 | %               | ~1 hours      | ~110M      |
| Mistral| %                | %               | ~6 hours      | ~7B        |
| Bard| %                | %               | ~1.5 hours      | ~110M        |

## Usage

### Converting Questions to SQL

```python
# Using BERT
from src.bert_model import generate_sql as bert_generate

question = "What is the population of New York?"
sql_query = bert_generate(model, question)

# Using Mistral
from src.mistral_model import generate_sql as mistral_generate

sql_query = mistral_generate(model, tokenizer, question)
```

## Dataset

The project uses the WikiSQL dataset, which contains:
- Natural language questions
- Corresponding SQL queries
- Table schemas

### Data Format Example:

```json
{
    "question": "How many people live in New York?",
    "sql": "SELECT population FROM city_data WHERE city = 'New York'",
    "table": {
        "name": "city_data",
        "columns": ["city", "population", "country"]
    }
}
```

## Performance Notes

- BERT Model:
  - Lighter weight, faster training
  - Good for simpler queries
  - More memory efficient

- Mistral Model:
  - Better understanding of complex queries
  - Requires more computational resources
  - Better generalization to unseen questions

## Future Improvements

1. Implement table schema awareness
2. Add support for more complex SQL operations
3. Create a web interface for demo purposes
4. Add query validation and safety checks
5. Improve handling of edge cases

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- WikiSQL dataset creators
- Hugging Face team for transformers library
- Mistral AI team for the Mistral-7B model
