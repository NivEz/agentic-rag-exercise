"""
Simple code-based grader using regex pattern matching.
"""

import sys
import re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluations.graders_dataset import questions, regex_patterns
from src.agents.query_router import QueryRouterAgent


def main():
    # Initialize router agent
    print("Initializing QueryRouterAgent...")
    router = QueryRouterAgent()
    print("Ready!\n")
    
    # Results tracking
    passed = 0
    failed = 0
    
    # Loop through each question
    for i, (question, pattern) in enumerate(zip(questions, regex_patterns), 1):
        print(f"\n[{i}/{len(questions)}] {question}")
        
        # Generate answer
        result = router.answer_with_contexts(question)
        answer = result['text_response']
        
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
    print(f"Results: {passed} passed, {failed} failed")
    print(f"Pass rate: {passed}/{len(questions)} ({passed/len(questions)*100:.1f}%)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
