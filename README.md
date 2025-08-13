# Clickbait Spoiling (SemEval 2023 Task)

This project addresses the two subtasks from the SemEval-2023 Clickbait Spoiling Challenge:

- **Task 1: Spoiler Type Classification** - Classify clickbait posts into three spoiler types: phrase, passage, or multi
- **Task 2: Spoiler Generation** - Generate spoiler text that satisfies curiosity induced by clickbait posts

Various models and training strategies were tested to effectively tackle both classification and generation tasks.

---

## Task 1: Spoiler Type Classification

The approach began with simple models (logistic regression) and gradually transitioned toward more powerful Transformer-based architectures.

**Final model:**
- Ensemble of `microsoft/deberta-v3-large` and `google/electra-large-discriminator`  
- Achieved **F1 score = 0.76**

---

## Task 2: Spoiler Generation

Encoder-decoder transformer models were used for generating spoilers from clickbait posts and linked articles.

**Final model:**
- Fine-tuned `t5-base`
- Achieved **METEOR score = 0.431**

*Note: Only the best model for both tasks are provided in this repo.*

---

## Dataset
The official JSONL files provided by the challenge organizers are used for this project:

train.jsonl
val.jsonl
test.jsonl
