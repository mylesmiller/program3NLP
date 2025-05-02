#!/usr/bin/env python3
from cs5322s25 import train_camper, train_conviction_model, train_deed_model

def main():
    print("Retraining models with expanded dataset...")
    
    print("\nTraining conviction model...")
    train_conviction_model()
    
    print("\nTraining camper model...")
    train_camper()
    
    print("\nTraining deed model...")
    train_deed_model()
    
    print("\nAll models have been retrained with the expanded dataset!")

if __name__ == "__main__":
    main() 