# GLiNER Romanian PII Anonymizer

## Description

GLiNER based Romanian PII data anonymizer

## How to Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python evaluator.py --data mock_subset_200.json --limit 200
```

## Experiment

4 list of labels

- en
- ro
- en_short
- ro_short

3 models

- urchade/gliner_multi_pii-v1 (pytorch.bin - 1.16GB)
- gliner-community/gliner_large-v2.5 (pytorch.bin - 1.84GB)
- knowledgator/gliner-x-large (pytorch.bin - 2.43GB)

## Results

| Model                              | Label    | Precision    | Recall       | F1           |
| ---------------------------------- | -------- | ------------ | ------------ | ------------ |
| urchade/gliner_multi_pii-v1        | en       | 0.7059942912 | 0.6052202284 | 0.6517347387 |
| urchade/gliner_multi_pii-v1        | ro       | 0.7281879195 | 0.530995106  | 0.6141509434 |
| knowledgator/gliner-x-large        | en       | 0.6720122184 | 0.7177814029 | 0.694143167  |
| knowledgator/gliner-x-large        | ro       | 0.6592510197 | 0.7251223491 | 0.6906195378 |
| gliner-community/gliner_large-v2.5 | en       | 0.4691176471 | 0.6504893964 | 0.545112782  |
| gliner-community/gliner_large-v2.5 | ro       | 0.4375382731 | 0.5827895595 | 0.4998251137 |
| urchade/gliner_multi_pii-v1        | en_short | 0.7620643432 | 0.4637030995 | 0.5765720081 |
| urchade/gliner_multi_pii-v1        | ro_short | 0.7777003484 | 0.4551386623 | 0.5742217649 |
| knowledgator/gliner-x-large        | en_short | 0.6362835755 | 0.589314845  | 0.6118992166 |
| knowledgator/gliner-x-large        | ro_short | 0.6457719815 | 0.6260195759 | 0.6357423897 |

## Other Options Considered

- BERT - needs fine-tuning
- Presidio with spaCy models - not very good with Romanian out of the box

## Future Iterations

1. Testing ModernBert based models (look promising)

   - also zero-shot
   - lighter/faster
   - bigger context

2. Finetuning gliner and/or modernerbert models
