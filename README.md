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

> _(To be added soon: test F1, precision, recall, etc.)_

---

## Quick Demo

```bash
$ python run_caee.py

Enter a sentence: I feel so left out and ignored.

Predicted Needs: ['comfort', 'support']
