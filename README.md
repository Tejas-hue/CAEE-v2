# CONTEXT AWARE EMPATHY ENGINE  
### v2.0 — The Transformer Awakens

---

## Why I Rebuilt This (Instead of Doing Literally Anything Else)

> Last time, I built a prototype that could take a sentence like “I feel lost” and tell you not just the emotion—but the underlying *need* behind it. Support, validation, comfort... things that go deeper than “happy” or “sad.”

> But, well
> v1 was... uhh... a patched-together system of sentence embeddings, custom mappings, and a glorified XGBoost model?

> It worked, and was ambitious...  
> But it wasn’t *contextually aware*. It didn’t really *get it*.

> So I went back to my table.  
> I taught myself HuggingFace transformers, read more research papers than I probably should’ve, and fine-tuned a DeBERTa-v3 model that can understand emotional subtext better than most humans I know.

> This is no longer just an emotion classifier, but now a contextual inference engine for psychological needs.

---

## What’s New in v2

> - Fully fine-tuned **DeBERTa-v3 Transformer**  
> - Supports **multi-label psychological need detection**  
> - Replaces SBERT + XGBoost pipeline with **end-to-end transformer inference**  
> - Built using **GoEmotions** dataset, mapped to a **curated hierarchy of needs**  
> - Added offline compatibility - **run everything locally**  
> - Refactored CLI script (`run_caee.py`) for faster, cleaner predictions  
> - Packaged with Git LFS for handling large model files  

---

## Who Might Want This

> - Mental health startups who actually care about nuance  
> - Chatbot developers tired of bots that just say “I’m sorry to hear that.”  
> - Researchers exploring context-aware affective computing  
> - Overambitious devs building emotionally intelligent agents  
> - That one friend designing an AI therapist instead of going to therapy (yes, still you)

---

## Under the Hood (v2 Edition)

> - **Backbone Model**: [`microsoft/deberta-v3-base`](https://huggingface.co/microsoft/deberta-v3-base)  
> - **Fine-Tuning Task**: Multi-label classification (`BCEWithLogitsLoss`)  
> - **Dataset**: Custom-mapped GoEmotions → psychological needs (CSV format)  
> - **Tokenization**: SentencePiece + padding + truncation (max length: 128)  
> - **Sigmoid Thresholding**: 0.3 cutoff for multi-label inference  
> - **Label Encoding**: `sklearn`'s `MultiLabelBinarizer` saved as `.pkl`  
> - **Frameworks**: `transformers`, `datasets`, `pandas`, `scikit-learn`, `joblib`, `PyTorch`  
> - **Model Size**: ~740MB `.safetensors`, Git LFS enabled  

---

## Model Performance

> CAEE-v2 was evaluated on a held-out validation set of 5,426 emotion-labeled sentences.
> The task was multi-label need classification (i.e., each sentence may imply multiple human needs simultaneously).

**Aggregate Metrics:**

- Micro F1-score: **0.65**
- Macro F1-score: **0.63**
- Sample-based F1-score: **0.65**
- Weighted F1-score: **0.65**

**Per-Need F1 Scores:**

| Need            | Precision | Recall | F1-score | Support |
|-----------------|-----------|--------|----------|---------|
| acknowledgment  | 0.81      | 0.87   | 0.84     | 306     |
| celebration     | 0.54      | 0.60   | 0.57     | 121     |
| clarity         | 0.53      | 0.69   | 0.60     | 522     |
| comfort         | 0.51      | 0.56   | 0.53     | 232     |
| connection      | 0.64      | 0.81   | 0.71     | 616     |
| motivation      | 0.52      | 0.48   | 0.50     | 211     |
| neutral         | 0.61      | 0.71   | 0.66     | 1592    |
| safety          | 0.59      | 0.60   | 0.60     | 85      |
| support         | 0.64      | 0.66   | 0.65     | 907     |
| understanding   | 0.53      | 0.72   | 0.61     | 75      |
| validation      | 0.57      | 0.68   | 0.62     | 759     |

> This evaluation reveals strengths in detecting common emotional needs (support, connection, acknowledgment), while also maintaining decent generalization across rarer ones like celebration and safety.


---

## Quick Demo

```bash
$ python run_caee.py

Enter a sentence: I feel so left out and ignored.

Predicted Needs: ['comfort', 'support']
