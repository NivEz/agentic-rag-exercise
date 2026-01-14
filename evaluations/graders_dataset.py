"""
Evaluation dataset with 20 factual questions from Insurance_Claim_Report_Comprehensive.pdf
Each entry contains question, regex pattern, and expected tool to be called.
"""

dataset = [
    # Numeric values (5 questions)
    {
        'question': 'What is the total amount of past medical expenses incurred by the claimant?',
        'regex_pattern': r'\$?\s*28,?450',
        'expected_tool': 'route_to_needle'
    },
    {
        'question': 'How many miles were on the odometer of the claimant\'s vehicle at the time of the accident?',
        'regex_pattern': r'18,?450\s*(miles)?',
        'expected_tool': 'route_to_needle'
    },
    {
        'question': 'What is the temperature in degrees Fahrenheit at the time of the accident?',
        'regex_pattern': r'52\s*(degrees|°)?\s*(F|Fahrenheit)?',
        'expected_tool': 'route_to_needle'
    },
    {
        'question': 'What is the actual cash value of the claimant\'s vehicle determined by the insurance company?',
        'regex_pattern': r'\$?\s*24,?500',
        'expected_tool': 'route_to_needle'
    },
    {
        'question': 'What is the weekly wage amount that the claimant earns as a construction supervisor?',
        'regex_pattern': r'\$?\s*1,?200\s*(per week|weekly|/week)?',
        'expected_tool': 'route_to_needle'
    },
    
    # Dates and times (4 questions)
    {
        'question': 'What is the exact time of the accident in 12-hour format with AM/PM?',
        'regex_pattern': r'2:47\s*(PM|p\.m\.|pm)',
        'expected_tool': 'route_to_needle'
    },
    {
        'question': 'What date was the police report filed?',
        'regex_pattern': r'October\s+15,?\s+2024',
        'expected_tool': 'route_to_needle'
    },
    {
        'question': 'What date was the claimant\'s statement taken by the adjuster?',
        'regex_pattern': r'October\s+16,?\s+2024',
        'expected_tool': 'route_to_needle'
    },
    {
        'question': 'What date was the vehicle inspection conducted at Springfield Auto Body?',
        'regex_pattern': r'October\s+17,?\s+2024',
        'expected_tool': 'route_to_needle'
    },
    
    # Identification numbers (3 questions)
    {
        'question': 'What is the policy number for the claimant\'s insurance?',
        'regex_pattern': r'POL-8472-9384-2931',
        'expected_tool': 'route_to_needle'
    },
    {
        'question': 'What is the badge number of the police officer who filed the report?',
        'regex_pattern': r'\b8472\b',
        'expected_tool': 'route_to_needle'
    },
    {
        'question': 'What is the official police report number?',
        'regex_pattern': r'SPR-2024-10-15-0847',
        'expected_tool': 'route_to_needle'
    },
    
    # Names and locations (3 questions)
    {
        'question': 'What is the license plate number of the claimant\'s Honda Accord?',
        'regex_pattern': r'IL-ABC-1234',
        'expected_tool': 'route_to_needle'
    },
    {
        'question': 'At what intersection did the accident occur?',
        'regex_pattern': r'Main\s+Street\s+and\s+Oak\s+Avenue',
        'expected_tool': 'route_to_needle'
    },
    {
        'question': 'What is the color description of the claimant\'s vehicle?',
        'regex_pattern': r'Midnight Blue Metallic',
        'expected_tool': 'route_to_needle'
    },
    
    # Additional questions (5 more)
    {
        'question': 'What is the total proposed settlement amount for this claim?',
        'regex_pattern': r'\$?\s*134,?650',
        'expected_tool': 'route_to_needle'
    },
    {
        'question': 'What is the name of the assigned claims adjuster?',
        'regex_pattern': r'Sarah\s+(Elizabeth\s+)?Thompson',
        'expected_tool': 'route_to_needle'
    },
    {
        'question': 'What is the claimant\'s occupation?',
        'regex_pattern': r'construction\s+supervisor',
        'expected_tool': 'route_to_needle'
    },
    {
        'question': 'How many weeks of past lost wages is the claimant claiming?',
        'regex_pattern': r'(seven|7)\s+weeks?',
        'expected_tool': 'route_to_needle'
    },
    {
        'question': 'At what speed was the Ford F-150 traveling when approaching the intersection?',
        'regex_pattern': r'(45|forty-five)\s*(mph|miles per hour)',
        'expected_tool': 'route_to_needle'
    },
]
