# Clickbait Spoiling (SemEval 2023 Task)

This project addresses the two subtasks from the SemEval-2023 Clickbait Spoiling Challenge:

- **Task 1: Spoiler Type Classification** - Classify clickbait posts into three spoiler types: phrase, passage, or multi
- **Task 2: Spoiler Generation** - Generate spoiler text that satisfies curiosity induced by clickbait posts

We experimented with various models and training strategies to tackle both classification and generation tasks effectively.

---

## Task 1: Spoiler Type Classification

We started with simple models (logistic regression) and gradually moved toward more powerful Transformer-based architectures.

**Final model:**
- Ensemble of `microsoft/deberta-v3-large` and `google/electra-large-discriminator`  
- Achieved **F1 score = 0.76**

---

## Task 2: Spoiler Generation

We used encoder-decoder transformer models for generating spoilers from clickbait posts and linked articles.

**Final model:**
- Fine-tuned `t5-base`
- Achieved **METEOR score = 0.431**

*Note: We only provide our best model for both tasks.*

---

## Team Members
- Neshat Sharbatdar
- Majid Taherkhani