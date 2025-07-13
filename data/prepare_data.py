from datasets import load_dataset
from tokenizers import ByteLevelBPETokenizer
from pathlib import Path

# Configuration
DATASET_NAME = "wikitext"
DATASET_CONFIG = "wikitext-2-raw-v1"
VOCAB_SIZE = 16_000
MIN_FREQUENCY = 2
OUTPUT_DIR = Path(__file__).parent.parent / "artifacts"
TOKENIZER_FILE = OUTPUT_DIR / "tokenizer.json"

def train_tokenizer():
    """ 
    Downloads the WikiText-2 dataset and trains a BPE tokenizer.
    """
    print(f"Loading dataset: {DATASET_NAME} ({DATASET_CONFIG})")
    # 1. Load the dataset
    dataset = load_dataset(DATASET_NAME, DATASET_CONFIG, split="train")

    # 2. Create an iterator over the text, filtering empty strings
    def batch_iterator(batch_size=1000):
        for i in range(0, len(dataset), batch_size):
            batch = [text for text in dataset[i : i + batch_size]["text"] if text and not text.isspace()]
            if batch:
                yield batch

    # 3. Initialize the tokenizer
    tokenizer = ByteLevelBPETokenizer()

    # 4. Train the tokenizer
    print("Training BPE tokenizer...")
    tokenizer.train_from_iterator(
        batch_iterator(),
        vocab_size=VOCAB_SIZE,
        min_frequency=MIN_FREQUENCY,
        special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"],
    )

    # 5. Save the tokenizer
    OUTPUT_DIR.mkdir(exist_ok=True)
    tokenizer.save(str(TOKENIZER_FILE))
    print(f"Tokenizer saved to: {TOKENIZER_FILE}")

if __name__ == "__main__":
    train_tokenizer()
