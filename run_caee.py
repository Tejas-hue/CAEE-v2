import torch
from transformers import DebertaV2Tokenizer, DebertaV2ForSequenceClassification
import joblib
import numpy as np
import os


model_dir = os.path.join("deberta_needs")


tokenizer = DebertaV2Tokenizer.from_pretrained(model_dir)
model = DebertaV2ForSequenceClassification.from_pretrained(model_dir)
model.eval()


mlb_path = os.path.join(model_dir, "label_encoder.pkl")
mlb = joblib.load(mlb_path)


def predict(text, threshold=0.3):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.sigmoid(logits).cpu().numpy()[0]
        predicted_indices = np.where(probs >= threshold)[0]
        predicted_needs = mlb.classes_[predicted_indices]
        return tuple(predicted_needs) if len(predicted_needs) > 0 else ("no strong need detected",)


def main():
    print("\n📣 Context-Aware Empathy Engine v2.0 — Interactive Mode")
    print("Type a sentence to analyze, or type 'exit' to quit.\n")

    while True:
        text = input("💬 You: ")
        if text.lower() in ["exit", "quit"]:
            print("👋 Exiting. Stay kind.")
            break
        result = predict(text)
        print(f"🤖 Predicted needs: {result}\n")

if __name__ == "__main__":
    main()
