#!/usr/bin/env python3
from cs5322s25 import WSD_Test_conviction, WSD_Test_camper, WSD_Test_deed

def test_model(name, test_func, sentences, expected_labels):
    """Test a model and print results."""
    print(f"\nTesting {name} model:")
    print("-" * 50)
    
    predictions = test_func(sentences)
    correct = sum(1 for pred, exp in zip(predictions, expected_labels) if pred == exp)
    accuracy = correct / len(sentences) * 100
    
    for i, (sentence, pred, exp) in enumerate(zip(sentences, predictions, expected_labels), 1):
        result = "✓" if pred == exp else "✗"
        print(f"\n{i}. {result} Predicted: {pred}, Expected: {exp}")
        print(f"   Sentence: {sentence}")
    
    print(f"\nAccuracy: {accuracy:.1f}% ({correct}/{len(sentences)})")

def main():
    # Test sentences for conviction
    conviction_sentences = [
        "The pianist played each note with such conviction that the audience felt a strong sense of emotion.",  # 1
        "Despite mounting criticism, she remained steadfast in her conviction that kindness has the power to change the world.",  # 1
        "No matter how hard they tried, they couldn't shake his conviction that LeBron James is the best basketball player of all time.",  # 1
        "DNA evidence proved pivotal in securing the suspect's conviction for the series of armed robberies.",  # 2
        "Having fifteen prior convictions on record, she knew her sentence would be severe.",  # 2
        "The conviction of the CEO for embezzlement sent shockwaves through the financial world."  # 2
    ]
    conviction_expected = [1, 1, 1, 2, 2, 2]

    # Test sentences for camper
    camper_sentences = [
        "The camper woke up early to watch the sunrise over the lake.",  # 1
        "The counselors took their campers on a trip to the lake as a reward for their good behavior.",  # 1
        "Each camper brought their own sleeping bag and flashlight.",  # 1
        "They traveled across the country in a fully equipped camper.",  # 2
        "The new camper has a tiny kitchen, bed, and even a small shower.",  # 2
        "Jane debated between staying in her camper and moving to a hotel for winter since it was supposed to start snowing."  # 2
    ]
    camper_expected = [1, 1, 1, 2, 2, 2]

    # Test sentences for deed
    deed_sentences = [
        "Before handing over the keys, the realtor carefully reviewed the deed with the new homeowners.",  # 1
        "The bank required an official copy of the deed before approving the construction loan.",  # 1
        "After his long lost Uncle Ben had passed away, his nephew received a deed to the mansion in the mail.",  # 1
        "His selfless deed of donating blood saved countless lives after the natural disaster.",  # 2
        "Superman is known for his good deed's done for the city of Megatown.",  # 2
        "While it feels self rewarding, the deed of community service is often unseen by others."  # 2
    ]
    deed_expected = [1, 1, 1, 2, 2, 2]

    # Test all models
    test_model("conviction", WSD_Test_conviction, conviction_sentences, conviction_expected)
    test_model("camper", WSD_Test_camper, camper_sentences, camper_expected)
    test_model("deed", WSD_Test_deed, deed_sentences, deed_expected)

if __name__ == "__main__":
    main() 