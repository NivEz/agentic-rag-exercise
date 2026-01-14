"""
Simple code-based grader using regex pattern matching and LLM-as-judge.
"""

import sys
import re
import os
import json
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluations.graders_dataset import dataset
from src.agents.query_router import QueryRouterAgent

load_dotenv()


def llm_judge(question: str, answer: str, ground_truth: str) -> dict:
    """
    Use LLM as a judge to evaluate answer quality.
    
    Args:
        question: The question asked
        answer: Generated answer from RAG system
        ground_truth: Expected answer
        
    Returns:
        Dictionary with 'score' (1-10) and 'feedback' (string)
    """
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    system_prompt = """You are an expert evaluator assessing the quality of answers from a RAG system.

Your task is to compare a generated answer against a ground truth reference and rate the quality.

Scoring criteria (1-10):
- 10: Perfect answer, includes all key information from ground truth
- 8-9: Excellent answer, includes most key information with minor omissions
- 6-7: Good answer, includes core information but missing some details
- 4-5: Adequate answer, partially correct but missing significant details
- 2-3: Poor answer, major inaccuracies or missing most information
- 1: Incorrect or completely off-topic answer

Provide your evaluation as JSON with:
{
    "score": <integer 1-10>,
    "feedback": "<explanation of score, what was good/missing>"
}"""
    
    user_prompt = f"""Question: {question}

Ground Truth (Expected Answer):
{ground_truth}

Generated Answer (To Evaluate):
{answer}

Evaluate the generated answer against the ground truth. Provide score and feedback in JSON format."""
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        return {
            'score': result.get('score', 0),
            'feedback': result.get('feedback', 'No feedback provided')
        }
    except Exception as e:
        return {
            'score': 0,
            'feedback': f'Error during evaluation: {str(e)}'
        }


def main():
    # Initialize router agent
    print("Initializing QueryRouterAgent...")
    router = QueryRouterAgent()
    print("Ready!\n")
    
    # Results tracking
    passed = 0
    failed = 0
    routing_correct = 0
    routing_incorrect = 0
    llm_judge_scores = []
    llm_judge_count = 0
    
    # Loop through each question
    for i, item in enumerate(dataset, 1):
        question = item['question']
        pattern = item['regex_pattern']
        expected_tool = item['expected_tool']
        
        print(f"\n[{i}/{len(dataset)}] {question}")
        
        # Generate answer
        result = router.answer_with_contexts(question)
        answer = result['text_response']
        tool = result.get('tool', 'unknown')
        
        # Check routing
        routing_status = ""
        if tool == expected_tool:
            routing_status = "✓"
            routing_correct += 1
        else:
            routing_status = "✗"
            routing_incorrect += 1
        
        # Display which tool was used
        print(f"{routing_status} Tool: {tool} (expected: {expected_tool})")
        
        # Check if pattern matches
        match = re.search(pattern, answer, re.IGNORECASE)
        
        if match:
            print(f"✓ PASS - Found: {match.group(0)}")
            passed += 1
        else:
            print(f"✗ FAIL - Pattern not found in answer")
            print(f"  Answer: {answer[:100]}...")
            failed += 1
        
        # Run LLM judge if ground_truth is available
        if 'ground_truth' in item:
            print(f"🤖 Running LLM Judge...")
            judge_result = llm_judge(question, answer, item['ground_truth'])
            print(f"   Score: {judge_result['score']}/10")
            print(f"   Feedback: {judge_result['feedback']}")
            llm_judge_scores.append(judge_result['score'])
            llm_judge_count += 1
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"ANSWER VALIDATION (Regex):")
    print(f"  Passed: {passed}/{len(dataset)} ({passed/len(dataset)*100:.1f}%)")
    print(f"  Failed: {failed}/{len(dataset)}")
    print(f"\nROUTING VALIDATION:")
    print(f"  Correct: {routing_correct}/{len(dataset)} ({routing_correct/len(dataset)*100:.1f}%)")
    print(f"  Incorrect: {routing_incorrect}/{len(dataset)}")
    
    if llm_judge_count > 0:
        avg_score = sum(llm_judge_scores) / len(llm_judge_scores)
        print(f"\nLLM JUDGE EVALUATION:")
        print(f"  Questions evaluated: {llm_judge_count}")
        print(f"  Average score: {avg_score:.1f}/10")
        print(f"  Score distribution:")
        for score in range(1, 11):
            count = llm_judge_scores.count(score)
            if count > 0:
                bar = "█" * count
                print(f"    {score:2d}: {bar} ({count})")
    
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
