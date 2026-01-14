"""
Evaluation dataset with 15 factual questions from Insurance_Claim_Report_Comprehensive.pdf
Each question has a corresponding regex pattern to validate the answer.
"""

questions = [
    # Numeric values (5 questions)
    'What is the total amount of past medical expenses incurred by the claimant?',
    'How many miles were on the odometer of the claimant\'s vehicle at the time of the accident?',
    'What is the temperature in degrees Fahrenheit at the time of the accident?',
    'What is the actual cash value of the claimant\'s vehicle determined by the insurance company?',
    'What is the weekly wage amount that the claimant earns as a construction supervisor?',
    
    # Dates and times (4 questions)
    'What is the exact time of the accident in 12-hour format with AM/PM?',
    'What date was the police report filed?',
    'What date was the claimant\'s statement taken by the adjuster?',
    'What date was the vehicle inspection conducted at Springfield Auto Body?',
    
    # Identification numbers (3 questions)
    'What is the policy number for the claimant\'s insurance?',
    'What is the badge number of the police officer who filed the report?',
    'What is the official police report number?',
    
    # Names and locations (3 questions)
    'What is the license plate number of the claimant\'s Honda Accord?',
    'At what intersection did the accident occur?',
    'What is the color description of the claimant\'s vehicle?',
    
    # Additional questions (5 more)
    'What is the total proposed settlement amount for this claim?',
    'What is the name of the assigned claims adjuster?',
    'What is the claimant\'s occupation?',
    'How many weeks of past lost wages is the claimant claiming?',
    'At what speed was the Ford F-150 traveling when approaching the intersection?',
]

regex_patterns = [
    # Numeric values
    r'\$?\s*28,?450',
    r'18,?450\s*(miles)?',
    r'52\s*(degrees|°)?\s*(F|Fahrenheit)?',
    r'\$?\s*24,?500',
    r'\$?\s*1,?200\s*(per week|weekly|/week)?',
    
    # Dates and times
    r'2:47\s*(PM|p\.m\.|pm)',
    r'October\s+15,?\s+2024',
    r'October\s+16,?\s+2024',
    r'October\s+17,?\s+2024',
    
    # Identification numbers
    r'POL-8472-9384-2931',
    r'\b8472\b',
    r'SPR-2024-10-15-0847',
    
    # Names and locations
    r'IL-ABC-1234',
    r'Main\s+Street\s+and\s+Oak\s+Avenue',
    r'Midnight Blue Metallic',
    
    # Additional patterns
    r'\$?\s*134,?650',
    r'Sarah\s+(Elizabeth\s+)?Thompson',
    r'construction\s+supervisor',
    r'(seven|7)\s+weeks?',
    r'(45|forty-five)\s*(mph|miles per hour)',
]
