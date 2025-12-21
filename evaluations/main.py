import os
from datetime import datetime
from ragas import evaluate
from datasets import Dataset
from ragas.metrics import answer_correctness, faithfulness, context_precision
from ragas.llms import llm_factory
from ragas.embeddings.openai_provider import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

# Setup OpenAI LLM - use AsyncOpenAI for async operations
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable is not set")

from openai import AsyncOpenAI
client = AsyncOpenAI(api_key=api_key)

# Setup embeddings (required for answer_correctness semantic similarity)
# Use OpenAIEmbeddings with async client
embeddings = OpenAIEmbeddings(client=client, model="text-embedding-3-small")

data_samples = {
    'question': ['When was the first super bowl?', 'Who won the most super bowls?'],
    'answer': ['The first superbowl was held on Jan 15, 1967', 'The most super bowls have been won by The New England Patriots'],
    'retrieved_contexts': [['The first superbowl was held on January 15, 1967', 'The New England Patriots have won the Super Bowl a record six times'], ['The first superbowl was held on January 15, 1967', 'The New England Patriots have won the Super Bowl a record six times']],
    'ground_truth': ['The first superbowl was held on January 15, 1967', 'The New England Patriots have won the Super Bowl a record six times']
}

dataset = Dataset.from_dict(data_samples)
score = evaluate(dataset,metrics=[answer_correctness, context_precision, faithfulness], embeddings=embeddings)

# Convert to pandas DataFrame and save to CSV
df = score.to_pandas()
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = "evaluations/results"
os.makedirs(output_dir, exist_ok=True)
output_file = f"{output_dir}/{timestamp}.csv"
df.to_csv(output_file, index=False)
print(f"\nResults saved to {output_file}")