import os
import sys
import asyncio
from pathlib import Path
from datetime import datetime
from ragas import aevaluate
from datasets import Dataset
from ragas.metrics import answer_correctness, faithfulness, context_precision
from ragas.llms import llm_factory
from ragas.embeddings.openai_provider import OpenAIEmbeddings
from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

load_dotenv()

# Setup OpenAI LLM - use AsyncOpenAI for async operations
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable is not set")

from openai import AsyncOpenAI
client = AsyncOpenAI(api_key=api_key)

# Setup embeddings (required for answer_correctness semantic similarity)
embeddings = OpenAIEmbeddings(client=client, model="text-embedding-3-small")

# Import the dataset
from evaluations.dataset import insurance_claim_dataset

# Import QueryRouterAgent
from src.agents.query_router import QueryRouterAgent


# Main execution
async def main():
    """Main async function to run the evaluation."""
    # Generate answers and retrieved contexts synchronously
    questions = insurance_claim_dataset['question']
    ground_truths = insurance_claim_dataset['ground_truth']
    
    print("Generating answers and retrieved contexts via QueryRouterAgent...")
    router_agent = QueryRouterAgent()
    
    answers = []
    retrieved_contexts = []
    
    # Process questions synchronously
    for i, question in enumerate(questions, 1):
        print(f"Processing question {i}/{len(questions)}: {question[:60]}...")
        result = router_agent.answer_with_contexts(question)
        answers.append(result['text_response'])
        retrieved_contexts.append(result['contexts'])
    
    # Create data_samples dictionary for RAGAS evaluation
    data_samples = {
        'question': questions,
        'answer': answers,
        'retrieved_contexts': retrieved_contexts,
        'ground_truth': ground_truths
    }
    
    dataset = Dataset.from_dict(data_samples)
    
    # Use async evaluate
    print("Running RAGAS evaluation (async)...")
    score = await aevaluate(dataset, metrics=[answer_correctness, context_precision, faithfulness], embeddings=embeddings)
    
    # Convert to pandas DataFrame and save to CSV
    df = score.to_pandas()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = "evaluations/results"
    os.makedirs(output_dir, exist_ok=True)
    output_file = f"{output_dir}/{timestamp}.csv"
    df.to_csv(output_file, index=False)
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    asyncio.run(main())