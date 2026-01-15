"""
Simple code-based grader using regex pattern matching and LLM-as-judge.
"""

import sys
import re
import os
import json
import csv
from pathlib import Path
from datetime import datetime
from litellm import completion
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluations.graders_dataset import dataset
from src.agents.query_router import QueryRouterAgent

load_dotenv()


def human_eval(question: str, answer: str) -> dict:
    """
    Prompt human evaluator for score and feedback.
    
    Args:
        question: The question asked
        answer: Generated answer from RAG system
        
    Returns:
        Dictionary with 'score' (1-10) and 'feedback' (string)
    """
    print("\n" + "="*60)
    print("👤 HUMAN EVALUATION REQUIRED")
    print("="*60)
    print(f"\nQuestion: {question}")
    print(f"\nAnswer: {answer}")
    print("\n" + "-"*60)
    
    # Get score
    while True:
        try:
            score_input = input("Enter your score (1-10): ").strip()
            score = int(score_input)
            if 1 <= score <= 10:
                break
            else:
                print("❌ Please enter a number between 1 and 10.")
        except ValueError:
            print("❌ Invalid input. Please enter a number between 1 and 10.")
    
    # Get feedback (required)
    print("\nEnter your feedback (required):")
    while True:
        feedback = input("Feedback: ").strip()
        if feedback:
            break
        else:
            print("❌ Feedback is required. Please provide your evaluation feedback.")
    
    print("="*60 + "\n")
    
    return {
        'score': score,
        'feedback': feedback
    }


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
        response = completion(
            model="gemini/gemini-2.5-flash",
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
    human_eval_scores = []
    human_eval_count = 0
    
    # Store detailed results for CSV export
    results_data = []
    
    # Loop through each question
    for i, item in enumerate(dataset, 1):
        question = item['question']
        pattern = item.get('regex_pattern')  # Optional
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
        
        # Initialize result record
        result_record = {
            'question_number': i,
            'question': question,
            'answer': answer,
            'expected_tool': expected_tool,
            'actual_tool': tool,
            'routing_correct': tool == expected_tool,
            'regex_pattern': pattern if pattern else 'N/A',
            'regex_passed': None,
            'llm_judge_score': None,
            'llm_judge_feedback': None,
            'human_eval_score': None,
            'human_eval_feedback': None
        }
        
        # Check if pattern matches (only if pattern is provided)
        if pattern:
            match = re.search(pattern, answer, re.IGNORECASE)
            
            if match:
                print(f"✓ PASS - Found: {match.group(0)}")
                passed += 1
                result_record['regex_passed'] = True
                result_record['regex_match'] = match.group(0)
            else:
                print(f"✗ FAIL - Pattern not found in answer")
                print(f"  Answer: {answer[:100]}...")
                failed += 1
                result_record['regex_passed'] = False
                result_record['regex_match'] = None
        else:
            print(f"⊘ No regex validation (summary question)")
        
        # Run LLM judge if ground_truth is available
        if 'ground_truth' in item:
            print(f"🤖 Running LLM Judge...")
            judge_result = llm_judge(question, answer, item['ground_truth'])
            print(f"   Score: {judge_result['score']}/10")
            print(f"   Feedback: {judge_result['feedback']}")
            llm_judge_scores.append(judge_result['score'])
            llm_judge_count += 1
            result_record['llm_judge_score'] = judge_result['score']
            result_record['llm_judge_feedback'] = judge_result['feedback']
        
        # Run human evaluation if marked for human eval
        if item.get('human_eval', False):
            eval_result = human_eval(question, answer)
            print(f"✓ Human evaluation recorded: {eval_result['score']}/10")
            human_eval_scores.append(eval_result['score'])
            human_eval_count += 1
            result_record['human_eval_score'] = eval_result['score']
            result_record['human_eval_feedback'] = eval_result['feedback']
        
        # Add result to data collection
        results_data.append(result_record)
    
    # Print summary
    regex_total = passed + failed
    print(f"\n{'='*60}")
    print(f"ANSWER VALIDATION (Regex):")
    if regex_total > 0:
        print(f"  Passed: {passed}/{regex_total} ({passed/regex_total*100:.1f}%)")
        print(f"  Failed: {failed}/{regex_total}")
        print(f"  Skipped: {len(dataset) - regex_total} (no pattern)")
    else:
        print(f"  No regex validations performed")
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
    
    if human_eval_count > 0:
        avg_score = sum(human_eval_scores) / len(human_eval_scores)
        print(f"\nHUMAN EVALUATION:")
        print(f"  Questions evaluated: {human_eval_count}")
        print(f"  Average score: {avg_score:.1f}/10")
        print(f"  Score distribution:")
        for score in range(1, 11):
            count = human_eval_scores.count(score)
            if count > 0:
                bar = "█" * count
                print(f"    {score:2d}: {bar} ({count})")
    
    print(f"{'='*60}")
    
    # Save results to CSV using built-in csv module
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = "evaluations/results"
    os.makedirs(output_dir, exist_ok=True)
    output_file = f"{output_dir}/grader_results_{timestamp}.csv"
    
    # Define CSV column headers
    fieldnames = [
        'question_number', 'question', 'answer', 'expected_tool', 'actual_tool',
        'routing_correct', 'regex_pattern', 'regex_passed', 'regex_match',
        'llm_judge_score', 'llm_judge_feedback', 'human_eval_score', 'human_eval_feedback'
    ]
    
    # Write to CSV
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results_data)
        
        # Add blank row before summary
        writer.writerow({})
        
        # Add summary statistics
        writer.writerow({'question_number': 'SUMMARY STATISTICS', 'question': '', 'answer': ''})
        writer.writerow({})
        
        # Regex validation summary
        regex_total = passed + failed
        writer.writerow({'question_number': 'ANSWER VALIDATION (Regex)', 'question': '', 'answer': ''})
        if regex_total > 0:
            writer.writerow({'question_number': 'Passed', 'question': f"{passed}/{regex_total} ({passed/regex_total*100:.1f}%)"})
            writer.writerow({'question_number': 'Failed', 'question': f"{failed}/{regex_total}"})
            writer.writerow({'question_number': 'Skipped', 'question': f"{len(dataset) - regex_total} (no pattern)"})
        else:
            writer.writerow({'question_number': 'No regex validations performed', 'question': ''})
        writer.writerow({})
        
        # Routing validation summary
        writer.writerow({'question_number': 'ROUTING VALIDATION', 'question': '', 'answer': ''})
        writer.writerow({'question_number': 'Correct', 'question': f"{routing_correct}/{len(dataset)} ({routing_correct/len(dataset)*100:.1f}%)"})
        writer.writerow({'question_number': 'Incorrect', 'question': f"{routing_incorrect}/{len(dataset)}"})
        writer.writerow({})
        
        # LLM judge summary
        if llm_judge_count > 0:
            avg_score = sum(llm_judge_scores) / len(llm_judge_scores)
            writer.writerow({'question_number': 'LLM JUDGE EVALUATION', 'question': '', 'answer': ''})
            writer.writerow({'question_number': 'Questions evaluated', 'question': str(llm_judge_count)})
            writer.writerow({'question_number': 'Average score', 'question': f"{avg_score:.1f}/10"})
            writer.writerow({'question_number': 'Score distribution', 'question': ''})
            for score in range(1, 11):
                count = llm_judge_scores.count(score)
                if count > 0:
                    writer.writerow({'question_number': f"  Score {score}", 'question': str(count)})
            writer.writerow({})
        
        # Human evaluation summary
        if human_eval_count > 0:
            avg_score = sum(human_eval_scores) / len(human_eval_scores)
            writer.writerow({'question_number': 'HUMAN EVALUATION', 'question': '', 'answer': ''})
            writer.writerow({'question_number': 'Questions evaluated', 'question': str(human_eval_count)})
            writer.writerow({'question_number': 'Average score', 'question': f"{avg_score:.1f}/10"})
            writer.writerow({'question_number': 'Score distribution', 'question': ''})
            for score in range(1, 11):
                count = human_eval_scores.count(score)
                if count > 0:
                    writer.writerow({'question_number': f"  Score {score}", 'question': str(count)})
    
    print(f"\n✓ Results saved to: {output_file}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
