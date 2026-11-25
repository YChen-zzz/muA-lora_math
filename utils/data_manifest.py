"""Check tokenized train/eval data identity across search trials."""
import fcntl
import hashlib
import json
from pathlib import Path


def dataset_digest(dataset):
    digest = hashlib.sha256()
    tokens = 0
    for row in dataset:
        if 'input_ids' not in row:
            raise ValueError('Token identity check requires prepared input_ids')
        selected = {k: row[k] for k in ['input_ids', 'attention_mask', 'assistant_masks',
                                       'completion_mask', 'labels'] if k in row}
        digest.update(json.dumps(selected, sort_keys=True, separators=(',', ':')).encode())
        digest.update(b'\n')
        tokens += len(row['input_ids'])
    return {'rows':len(dataset), 'tokens':tokens, 'sha256':digest.hexdigest()}


def verify_data(trainer, args):
    global_batch = args.batch_size * args.grad_accum * trainer.args.world_size
    examples_per_epoch = len(trainer.train_dataset)
    if args.epochs < 1 or examples_per_epoch % global_batch:
        raise ValueError('Every epoch must contain an exact number of global batches')
    expected_steps = (examples_per_epoch // global_batch) * args.epochs
    if args.max_steps not in (-1, expected_steps) or trainer.args.dataloader_drop_last:
        raise ValueError('Fixed-data search must consume every example exactly once per epoch')
    train = dataset_digest(trainer.train_dataset)
    contract = {'seed':args.seed, 'max_seq_length':args.max_seq_length,
                'eval_batch_size':args.eval_batch_size,
                'assistant_only_loss':True,
                'epochs':args.epochs,
                'total_example_visits':examples_per_epoch * args.epochs,
                'total_sequence_tokens':train['tokens'] * args.epochs,
                'train':train,
                'eval':dataset_digest(trainer.eval_dataset)}
    path = Path(args.data_manifest)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix('.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if path.exists():
            if json.loads(path.read_text()) != contract:
                raise ValueError(f'Train/eval token or loss-mask mismatch against {path}')
        else:
            path.write_text(json.dumps(contract, indent=2)+'\n')
    (Path(args.save_dir)/'data_manifest.json').write_text(json.dumps(contract, indent=2)+'\n')
    print('Verified train/eval token and mask identity:', contract, flush=True)
