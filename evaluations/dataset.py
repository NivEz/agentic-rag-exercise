"""
Dataset containing questions and ground truth answers for the insurance claim report evaluation.
Based on the comprehensive insurance claim report: Insurance_Claim_Report_Comprehensive.pdf
"""

insurance_claim_dataset = {
    'question': [
        'What is the claim number for this insurance claim?',
        'What was the date and time of the accident?',
        'Where did the accident occur?',
        'Who is the claimant in this case?',
        'What type of vehicle was the claimant driving and what was its license plate number?',
        'Who was determined to be primarily liable for the accident and why?',
        'What were the weather conditions at the time of the accident?',
        'What injuries did the claimant sustain as a result of the accident?',
        'What is the total proposed settlement amount for this claim?',
        'How many witnesses provided statements about the accident and what were their names?'
    ],
    'ground_truth': [
        'The claim number is CLM-2024-0847-2931.',
        'The accident occurred on October 15, 2024 at approximately 2:47 PM Central Standard Time.',
        'The accident occurred at the intersection of Main Street and Oak Avenue in downtown Springfield, Illinois.',
        'The claimant is James Robert Mitchell.',
        'The claimant was driving a 2022 Honda Accord, four-door sedan, color Midnight Blue Metallic, with license plate IL-ABC-1234.',
        'The driver of the Ford F-150 pickup truck, Robert William Johnson, was determined to be primarily liable for the accident because he failed to yield at a traffic signal, entering the intersection against a red light, and was operating at an unsafe speed for conditions.',
        'At the time of the accident, there was light rain that had been falling for approximately thirty minutes, creating wet road surfaces. Visibility was reduced due to overcast skies and moderate precipitation. The temperature was approximately 52 degrees Fahrenheit.',
        'The claimant sustained cervical strain and sprain, lumbar strain and sprain, mild traumatic brain injury with post-concussion syndrome, and multiple contusions and abrasions.',
        'The total proposed settlement amount is $134,650, which includes property damage, medical expenses, lost wages, and pain and suffering compensation.',
        'Three witnesses provided statements about the accident: Maria Elena Rodriguez, David Michael Anderson, and Jennifer Lynn Martinez.'
    ]
}


