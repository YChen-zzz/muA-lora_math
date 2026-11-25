import argparse
import json
import logging
import os
from typing import Union, List
from unittest.mock import patch

from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainerCallback, set_seed
from transformers.utils import import_utils

if os.environ.get("MUA_SFT_DISABLE_VLLM") == "1":
    package_available = import_utils._is_package_available

    def sft_package_available(package, return_version=False):
        if package == "vllm":
            return (False, "N/A") if return_version else False
        return package_available(package, return_version=return_version)

    with patch.object(import_utils, "_is_package_available", sft_package_available):
        from trl import SFTConfig, SFTTrainer
else:
    from trl import SFTConfig, SFTTrainer

import wandb

from utils.helper import setup_logging, naming_conversion, random_split
from utils.lora_utils import custom_init
from utils.data_utils import filter_by_length
from utils.data_manifest import verify_data


class EvaluateAtEndCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step >= state.max_steps:
            control.should_evaluate = True
        return control


def load_peft_llam3(base_model_name: str,
                    rank: int,
                    target_modules : Union[List[str], str] = 'all-linear',
                    lora_dropout: float = 0.0,
                    init_lora_weights: bool = True,
                    mode: str = None):
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name, torch_dtype="float32" if rank == 0 else "auto"
    )
    base_model.config.use_cache = False

    if rank == 0:
        return base_model

    if mode is not None and 'alpha1' in mode:
        lora_alpha = 1
        use_rslora = False
    elif mode is not None and 'constant1' in mode:
        lora_alpha = rank
        use_rslora = False
    else:
        lora_alpha = 2 * rank
        use_rslora = False

    lora_config = LoraConfig(
        r=rank,
        lora_alpha=lora_alpha,
        use_rslora=use_rslora,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        init_lora_weights=init_lora_weights,
    )

    model = get_peft_model(base_model, lora_config)

    if mode is not None:
        if mode.startswith('initA'):
            custom_init(model, 'initA')
        elif mode.startswith('initB'):
            custom_init(model, 'initB')

    return model


def main(args):
    # Load tokenizer
    model_source = args.model_path or f"meta-llama/{args.base_model}"
    tokenizer = AutoTokenizer.from_pretrained(model_source)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    with open('chat_templates/llama_template.jinja', 'r') as f:
        tokenizer.chat_template = f.read()

    # Load dataset
    if args.dataset_dir:
        if args.filter_length:
            raise ValueError('The fixed MetaMathQA split cannot be filtered after selection')
        data_files = {
            'train': os.path.join(args.dataset_dir, 'train.jsonl'),
            'validation': os.path.join(args.dataset_dir, 'validation.jsonl'),
        }
        split = load_dataset('json', data_files=data_files)
        train_set = split['train'].select_columns(['messages'])
        test_set = split['validation'].select_columns(['messages'])
        if len(train_set) != 38912 or len(test_set) != 512:
            raise ValueError('Unexpected MetaMathQA split size')
    else:
        raw_data = load_dataset(f"allenai/{args.task}", split='train')
        if args.filter_length:
            raw_data = filter_by_length(raw_data, tokenizer, max_len=args.max_seq_length)
        train_indices, test_indices = random_split(len(raw_data), seed=args.seed)
        train_set = raw_data.select(train_indices[: 32 * args.train_set_size])
        test_set = raw_data.select(test_indices[: 32 * args.test_set_size])

    # create model
    model = load_peft_llam3(base_model_name=model_source,
                            rank=args.rank,
                            target_modules=args.target_modules,
                            lora_dropout=args.lora_dropout,
                            init_lora_weights=True,
                            mode=args.init_method)

    # training
    sft_config = SFTConfig(output_dir=args.save_dir,
                           per_device_train_batch_size=args.batch_size,
                           per_device_eval_batch_size=args.eval_batch_size,
                           gradient_accumulation_steps=args.grad_accum,
                           learning_rate=args.learning_rate,
                           weight_decay=args.weight_decay,
                           adam_beta1=args.adam_beta1,
                           adam_beta2=args.adam_beta2,
                           max_grad_norm=args.max_grad_norm,
                           num_train_epochs=args.epochs,
                           max_steps=args.max_steps,
                           warmup_ratio=args.warmup_ratio,
                           lr_scheduler_type=args.lr_scheduler_type,
                           lr_scheduler_kwargs=args.lr_scheduler_kwargs,
                           save_strategy='no' if args.benchmark or args.no_save else 'steps',
                           save_steps=args.save_steps,
                           save_only_model=False,
                           bf16=True,
                           tf32=True,
                           gradient_checkpointing=True,
                           gradient_checkpointing_kwargs={'use_reentrant': False},
                           dataloader_pin_memory=True,
                           dataloader_persistent_workers=True,
                           dataloader_num_workers=8,
                           dataloader_prefetch_factor=2,
                           logging_strategy='steps',
                           logging_steps=args.logging_steps,
                           eval_strategy='no' if args.benchmark else 'steps',
                           eval_steps=args.eval_steps,
                           report_to=['wandb'] if args.enable_log else [],
                           disable_tqdm=False,
                           seed=args.seed,
                           dataset_num_proc=16,
                           max_length=args.max_seq_length,
                           pad_to_multiple_of=8,
                           assistant_only_loss=True,)

    sft_trainer = SFTTrainer(model=model,
                             args=sft_config,
                             train_dataset=train_set,
                             eval_dataset=test_set,
                             processing_class=tokenizer,
                             callbacks=[] if args.benchmark else [EvaluateAtEndCallback()])

    if args.data_manifest:
        if sft_trainer.is_world_process_zero():
            verify_data(sft_trainer, args)
        sft_trainer.accelerator.wait_for_everyone()
    train_result = sft_trainer.train()
    if not args.benchmark:
        if not args.no_save:
            sft_trainer.save_model()
        sft_trainer.save_state()
        if sft_trainer.is_world_process_zero():
            final_step = sft_trainer.state.global_step
            last_train_log = next(
                (entry for entry in reversed(sft_trainer.state.log_history) if 'loss' in entry),
                {},
            )
            final_eval_loss = next(
                entry['eval_loss']
                for entry in reversed(sft_trainer.state.log_history)
                if entry.get('step') == final_step and 'eval_loss' in entry
            )
            with open(os.path.join(args.save_dir, 'final_metrics.json'), 'w', encoding='utf-8') as metrics_file:
                json.dump({
                    'global_step': final_step,
                    'learning_rate_exponent': args.learning_rate_exponent,
                    'peak_learning_rate': args.learning_rate,
                    'train_loss': train_result.training_loss,
                    'last_logged_train_loss': last_train_log.get('loss'),
                    'last_logged_train_loss_step': last_train_log.get('step'),
                    'eval_loss': final_eval_loss,
                }, metrics_file, indent=2)
                metrics_file.write('\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Basic arguments
    parser.add_argument('--base-model', type=str,
                        choices=['Llama-3.2-1B',],
                        default='Llama-3.2-1B')
    parser.add_argument('--task', type=str,
                        choices=['tulu-3-sft-mixture', 'metamathqa'],
                        default='tulu-3-sft-mixture')
    parser.add_argument('--model-path', type=str)
    parser.add_argument('--dataset-dir', type=str)

    # lora relevant
    parser.add_argument('--rank', type=int, required=True)
    parser.add_argument('--lora-dropout', type=float, default=0.0)
    parser.add_argument("--target-modules",
                        nargs="+",
                        metavar="MODULE",
                        default=['gate_proj', 'up_proj', 'down_proj'],)
    parser.add_argument('--init-method', type=str, default=None,
                        choices=[None, 'initA', 'initB', 'initA_alpha1', 'initB_alpha1', 'alpha1',
                                 'initA_constant1', 'initB_constant1', 'constant1'])

    # training relevant
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--max-steps', type=int, default=-1)
    parser.add_argument('--benchmark', action='store_true')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--eval-batch-size', type=int, default=32)
    parser.add_argument('--grad-accum', type=int, default=1)
    parser.add_argument('--learning-rate-exponent', type=float, required=True)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--adam-beta1', type=float, default=0.9)
    parser.add_argument('--adam-beta2', type=float, default=0.999)
    parser.add_argument('--max-grad-norm', type=float, default=1.0)
    parser.add_argument('--warmup-ratio', type=float, default=0.05)
    parser.add_argument('--lr-scheduler-type', type=str, default='cosine_with_min_lr')
    parser.add_argument('--min-lr-rate', type=float, default=0.1)
    parser.add_argument('--max-seq-length', type=int, default=1024)
    parser.add_argument('--save-steps', type=int, default=5000)
    parser.add_argument('--no-save', action='store_true')
    parser.add_argument('--output-dir', type=str)
    parser.add_argument('--eval-steps', type=int, default=500)
    parser.add_argument('--logging-steps', type=int, default=25)

    # training set size
    parser.add_argument('--train-set-size', type=int, default=10000)
    parser.add_argument('--test-set-size', type=int, default=1000)
    parser.add_argument('--filter-length', action='store_true')
    parser.add_argument('--data-manifest', type=str)

    # logging
    parser.add_argument('--enable-log', action='store_true')
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    args.learning_rate = 2 ** (-args.learning_rate_exponent)

    if args.init_method is not None and args.rank != 0:
        args.suffix = '-' + args.init_method
    else:
        args.suffix = ''
    if args.benchmark:
        args.suffix += '-benchmark'

    if args.lr_scheduler_type == 'cosine_with_min_lr':
        args.lr_scheduler_kwargs = {'min_lr_rate': args.min_lr_rate}
    else:
        args.lr_scheduler_kwargs = None

    save_dir = args.output_dir or os.path.join('checkpoints', args.base_model, args.task, f'LoRA-{args.rank}{args.suffix}', f'LR-{naming_conversion(args.learning_rate_exponent)}')
    os.makedirs(save_dir, exist_ok=True)

    log_dir = args.output_dir or os.path.join('logs', args.base_model, args.task, f'LoRA-{args.rank}{args.suffix}')
    setup_logging(log_dir)
    logging.info(args)

    args.log_dir = log_dir
    args.save_dir = save_dir
    set_seed(args.seed, deterministic=False)

    main(args)
