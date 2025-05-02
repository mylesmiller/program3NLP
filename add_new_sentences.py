#!/usr/bin/env python3

def add_sentences_to_file(file_path, new_sentences_sense1, new_sentences_sense2):
    """Add new sentences to a training data file."""
    with open(file_path, 'a') as f:
        # Add sense 1 sentences
        for sentence in new_sentences_sense1:
            f.write(f"1 {sentence}\n")
        
        # Add sense 2 sentences
        for sentence in new_sentences_sense2:
            f.write(f"2 {sentence}\n")

def main():
    # New sentences for conviction
    conviction_sense1 = [
        "The pianist played each note with such conviction that the audience felt a strong sense of emotion.",
        "Even with the odds stacked against them, the team played with the conviction that they could come back and win.",
        "Her inspiring conviction that education could transform lives drove her to open schools in rural villages.",
        "Despite mounting criticism, she remained steadfast in her conviction that kindness has the power to change the world.",
        "His lifelong commitment to education stemmed from a deep conviction in its transformative potential.",
        "The leader's words resonated with such conviction that even the most skeptical in the audience began to reconsider their views.",
        "She spoke with such conviction that everyone in the room believed her.",
        "His conviction in the power of education never wavered, even during hard times.",
        "Despite the criticism, he held onto his conviction that change was possible.",
        "No matter how hard they tried, they couldn't shake his conviction that LeBron James is the best basketball player of all time.",
        "Her strong conviction in environmentalism lead her to go completely vegan.",
        "My conviction that afternoon coffee is good for you isn't based on science — it's based on survival."
    ]

    conviction_sense2 = [
        "After a lengthy trial, the jury reached a unanimous decision, which consequently led to his conviction on all charges.",
        "Due to his prior conviction, he faced challenges when finding employment opportunities.",
        "The new evidence brought to light cast serious doubt on the legitimacy of the original conviction.",
        "DNA evidence proved pivotal in securing the suspect's conviction for the series of armed robberies.",
        "After three weeks of deliberation, the jury delivered a unanimous conviction on all counts.",
        "Years later, new forensic evidence emerged, leading to the overturning of his wrongful conviction.",
        "The court handed down a conviction after reviewing all the evidence.",
        "His previous convictions made it difficult for him to find a job.",
        "The conviction for theft resulted in a three-year prison sentence.",
        "The conviction came quickly after the jury reviewed the surveillance footage",
        "Having fifteen prior convictions on record, she knew her sentence would be severe.",
        "The conviction of the CEO for embezzlement sent shockwaves through the financial world."
    ]

    # New sentences for camper
    camper_sense1 = [
        "After spending a weekend camping in the forest, Sarah and her friends enjoyed the simplicity of living in a camper, making it their home away from home for a few days.",
        "The couple decided to explore the countryside, staying in a camper for a week to fully immerse themselves in nature.",
        "Each summer, dozens of camper families return to the same forest clearing to relive their favorite outdoor traditions",
        "The camper woke up early to watch the sunrise over the lake.",
        "Each camper brought their own sleeping bag and flashlight.",
        "The summer camp provided activities to keep every camper engaged.",
        "The counselors took their campers on a trip to the lake as a reward for their good behavior.",
        "The camper made sure to pack extra umbrellas and jackets in case it began to rain.",
        "The campers gathered around the fire when the sun began to set to stay warm."
    ]

    camper_sense2 = [
        "The family packed up their gear and hit the road in their camper, ready for an unforgettable summer road trip across the country.",
        "They decided to rent a camper for their vacation, allowing them to explore remote national parks while having all the comforts of home on wheels.",
        "After weeks of planning, Mark and his friends finally embarked on their cross-country adventure, traveling in a fully-equipped camper that had everything they needed for the journey.",
        "They traveled across the country in a fully equipped camper.",
        "The family parked their camper near the forest and set up for the night.",
        "The new camper has a tiny kitchen, bed, and even a small shower.",
        "The group of friends decided to rent a camper for their roadtrip though the mountains.",
        "Jane debated between staying in her camper and moving to a hotel for winter since it was supposed to start snowing.",
        "The camper was equip with a kitchen and a full size bed, but unfortunately did not have a shower."
    ]

    # New sentences for deed
    deed_sense1 = [
        "Before handing over the keys, the realtor carefully reviewed the deed with the new homeowners.",
        "The bank required an official copy of the deed before approving the construction loan.",
        "She inherited the cottage after discovering a forgotten deed buried in a stack of family papers.",
        "He formally executed the deed at the county recorder's office to transfer the farm into his mother's name.",
        "She stored the certified copy of the deed in her bank's safe deposit box for safekeeping.",
        "The deed clearly stipulates that the surviving spouse retains lifetime residency on the estate.",
        "The lawyer handed over the deed to finalize the sale of the house.",
        "They found the original deed in a locked drawer in the attic.",
        "Without the deed, she couldn't prove ownership of the farmland.",
        "After finalizing the mortgage, the bank handed me the deed to my new condo.",
        "The historical archive on the property included a hand-written deed from the 18th century.",
        "Before listing the estate, the lawyer verified that the deed was clear of any liens.",
        "After purchasing the property, she shortly received the completed deed via the mail.",
        "A Deed must be signed by all cooperating parties before the transfer of ownership finishes.",
        "After his long lost Uncle Ben had passed away, his nephew received a deed to the mansion in the mail."
    ]

    deed_sense2 = [
        "His selfless deed of donating blood saved countless lives after the natural disaster.",
        "They erected a monument to honor the soldier's heroic deed during the final battle.",
        "The stranger's simple deed of returning a lost wallet restored her faith in humanity.",
        "She organized a neighborhood cleanup, a charitable deed that brought the community closer together.",
        "In the midst of the crisis, his brave deed of guiding stranded passengers to safety earned him a commendation.",
        "They launched a tree-planting campaign in the park, an environmental deed that will benefit generations to come.",
        "Helping the elderly woman cross the street was a kind deed.",
        "He was honored for his heroic deed during the rescue mission.",
        "The community praised her for her selfless deeds throughout the year.",
        "Helping her elderly neighbor carry groceries was a small deed that brightened his day.",
        "Publishing the research paper with open access was a bold deed that advanced scientific collaboration.",
        "Donating supplies to the local shelter felt like the right deed on such a rainy afternoon.",
        "Superman is known for his good deed's done for the city of Megatown.",
        "No one will forget the deed done on Omaha Beach June 6th, 1944.",
        "While it feels self rewarding, the deed of community service is often unseen by others."
    ]

    # Add sentences to each file
    add_sentences_to_file('prog3/conviction.txt', conviction_sense1, conviction_sense2)
    add_sentences_to_file('prog3/camper.txt', camper_sense1, camper_sense2)
    add_sentences_to_file('prog3/deed.txt', deed_sense1, deed_sense2)

if __name__ == "__main__":
    main() 