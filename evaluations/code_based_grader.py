"""
Simple code-based grader using regex pattern matching.
"""

import sys
import re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluations.graders_dataset import dataset
from src.agents.query_router import QueryRouterAgent


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
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"ANSWER VALIDATION:")
    print(f"  Passed: {passed}/{len(dataset)} ({passed/len(dataset)*100:.1f}%)")
    print(f"  Failed: {failed}/{len(dataset)}")
    print(f"\nROUTING VALIDATION:")
    print(f"  Correct: {routing_correct}/{len(dataset)} ({routing_correct/len(dataset)*100:.1f}%)")
    print(f"  Incorrect: {routing_incorrect}/{len(dataset)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
