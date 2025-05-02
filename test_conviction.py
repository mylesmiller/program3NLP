#!/usr/bin/env python3
from cs5322s25 import WSD_Test_conviction

def main():
    # 1–4: your four sample sentences
    sentences = [
        "His bloody palm print on the bat eventually led to his conviction.",
        "A Sioux City woman found guilty of over 50 voter fraud charges is seeking to appeal her conviction.",
        "He spoke with conviction and sincerity.",
        "What are the grounds for appealing a conviction?"
    ]

    # call your WSD function
    preds = WSD_Test_conviction(sentences)

    # print one label per line
    for label in preds:
        print(label)

if __name__ == "__main__":
    main()
