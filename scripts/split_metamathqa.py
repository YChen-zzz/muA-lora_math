"""Create a fixed MetaMathQA train/validation split without question leakage."""

import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path('/data/250010186/hf/hub/datasets--meta-math--MetaMathQA/snapshots/aa4f34d3d2d3231299b5b03d9b3e5a20da45aa18/MetaMathQA-395K.json')
DEST = ROOT / 'data' / 'metamathqa_38912_train_512_val_seed42'
SEED = 42
TRAIN_COUNT = 38_912
VAL_COUNT = 512


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def write_rows(path, rows, indices):
    with path.open('w', encoding='utf-8') as stream:
        for index in indices:
            row = rows[index]
            item = {
                'source_index': index,
                'original_question': row['original_question'],
                'type': row['type'],
                'messages': [
                    {'role': 'user', 'content': row['query']},
                    {'role': 'assistant', 'content': row['response']},
                ],
            }
            stream.write(json.dumps(item, ensure_ascii=False) + '\n')


def main():
    if DEST.exists():
        raise FileExistsError(f'Refusing to replace existing split: {DEST}')
    rows = json.loads(SOURCE.read_text(encoding='utf-8'))
    by_question = defaultdict(list)
    for index, row in enumerate(rows):
        by_question[row['original_question']].append(index)
    if len(by_question) < VAL_COUNT:
        raise ValueError('Too few distinct original questions')

    rng = random.Random(SEED)
    val_questions = set(rng.sample(sorted(by_question), VAL_COUNT))
    val_indices = [rng.choice(by_question[question]) for question in sorted(val_questions)]
    train_candidates = [index for index, row in enumerate(rows)
                        if row['original_question'] not in val_questions]
    if len(train_candidates) < TRAIN_COUNT:
        raise ValueError('Too few non-overlapping training examples')
    train_indices = rng.sample(train_candidates, TRAIN_COUNT)
    if {rows[index]['original_question'] for index in train_indices} & val_questions:
        raise AssertionError('Question leakage detected')

    DEST.mkdir(parents=True)
    train_path = DEST / 'train.jsonl'
    val_path = DEST / 'validation.jsonl'
    write_rows(train_path, rows, train_indices)
    write_rows(val_path, rows, val_indices)
    manifest = {
        'source': str(SOURCE),
        'source_sha256': digest(SOURCE),
        'seed': SEED,
        'train_rows': TRAIN_COUNT,
        'validation_rows': VAL_COUNT,
        'train_unique_original_questions': len({rows[index]['original_question'] for index in train_indices}),
        'validation_unique_original_questions': VAL_COUNT,
        'excluded_source_rows_sharing_validation_questions': sum(len(by_question[q]) for q in val_questions) - VAL_COUNT,
        'train_type_counts': dict(Counter(rows[index]['type'] for index in train_indices)),
        'validation_type_counts': dict(Counter(rows[index]['type'] for index in val_indices)),
        'train_sha256': digest(train_path),
        'validation_sha256': digest(val_path),
    }
    (DEST / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
