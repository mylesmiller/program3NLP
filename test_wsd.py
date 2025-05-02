#!/usr/bin/env python3
import os
from cs5322s25 import WSD_Test_conviction, WSD_Test_camper, WSD_Test_deed

def process_test_file(word, test_file, output_file):
    """
    Process a test file through the appropriate WSD function and save results.
    
    Args:
        word (str): The target word ('camper', 'conviction', or 'deed')
        test_file (str): Path to the test file
        output_file (str): Path to save the results
    """
    # Read test sentences
    with open(test_file, 'r') as f:
        sentences = [line.strip() for line in f if line.strip()]
    
    # Select the appropriate WSD function
    wsd_functions = {
        'camper': WSD_Test_camper,
        'conviction': WSD_Test_conviction,
        'deed': WSD_Test_deed
    }
    
    if word not in wsd_functions:
        raise ValueError(f"Unsupported word: {word}")
    
    # Get predictions
    predictions = wsd_functions[word](sentences)
    
    # Save results
    with open(output_file, 'w') as f:
        for pred in predictions:
            f.write(f"{pred}\n")
    
    print(f"Processed {len(sentences)} sentences for {word}")
    print(f"Results saved to {output_file}")

def main():
    # Test files and corresponding words
    test_files = {
        'camper': 'camper_test.txt',
        'conviction': 'conviction_test.txt',
        'deed': 'deed_test.txt'
    }
    
    # Process each test file
    for word, test_file in test_files.items():
        if not os.path.exists(test_file):
            print(f"Warning: Test file {test_file} not found")
            continue
            
        # Create output filename with MylesMiller
        output_file = f"result_{word}_MylesMiller.txt"
        
        try:
            process_test_file(word, test_file, output_file)
        except Exception as e:
            print(f"Error processing {word}: {str(e)}")

if __name__ == "__main__":
    main() 