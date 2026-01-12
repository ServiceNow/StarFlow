from accelerate.utils import gather_object, tqdm
from accelerate.utils.modeling import load_checkpoint_in_model
from pathlib import Path
from torch import inference_mode
from torch.optim import AdamW
from torch.utils.data import ConcatDataset, DataLoader
from transformers import get_cosine_schedule_with_warmup
from starvlm.model.base import VLLocalModel
from starvlm.pipeline.utils import (
    get_accelerator,
    get_config_by_composition,
    get_logging_dir,
    LossMeter,
    load_checkpoint,
    save_checkpoint_and_export_model,
)
from starvlm.utils import get_class_instance
import argparse
import math
import time


@inference_mode()
def validate(vl_local_model: VLLocalModel, validate_data_loader: DataLoader) -> None:
    loss_meter = LossMeter()
    progress_bar = tqdm(desc="validate", total=len(validate_data_loader))
    for vl_local_input in validate_data_loader:
        loss = vl_local_model.forward(vl_local_input)
        loss_meter.update(loss.item(), vl_local_input.labels.shape[0])
        progress_bar.update()
    progress_bar.close()
    validate_loss = [loss_meter.average]
    validate_loss = gather_object(validate_loss)
    validate_loss = sum(validate_loss) / len(validate_loss)
    return validate_loss


def train(pipeline_name: str, model_name: str, dataset_names: list[str]) -> None:
    dataset_names = sorted(dataset_names)
    config = get_config_by_composition(
        {
            "pipeline": f"pipeline/{pipeline_name}",
            "model": f"model/{model_name}",
            "datasets": ",".join(
                f"dataset/{dataset_name}" for dataset_name in dataset_names
            ),
        }
    )
    logging_dir = get_logging_dir(
        f"{pipeline_name}/{model_name}/{'__'.join(dataset_names)}", config
    )
    vl_local_model = get_class_instance(
        config["model"]["class_path"], **config["model"]["init_kwargs"]
    )
    accelerator = get_accelerator(
        config["pipeline"]["accelerator_kwargs"],
        fsdp_kwargs=config["pipeline"]["fsdp_kwargs"],
        vl_local_model=vl_local_model,
        use_wandb=True,
        pipeline_name=pipeline_name,
        logging_dir=logging_dir,
    )
    accelerator.print(f"logging directory: '{logging_dir}'")
    accelerator.print(f"number of processes: {accelerator.num_processes}")
    accelerator.print(
        f"gradient accumulation steps: {accelerator.gradient_accumulation_steps}"
    )
    if config["pipeline"]["model_checkpoint"] is not None:
        load_checkpoint_in_model(
            vl_local_model.model, checkpoint=config["pipeline"]["model_checkpoint"]
        )
        accelerator.print(
            f"loaded model checkpoint from '{config['pipeline']['model_checkpoint']}'"
        )
    if config["model"]["coordinates_adapter"] is None:
        coordinates_adapter = None
    else:
        coordinates_adapter = get_class_instance(
            config["model"]["coordinates_adapter"]["class_path"],
            **config["model"]["coordinates_adapter"]["init_kwargs"],
        )
    train_vl_datasets = []
    for dataset_config in config["datasets"]:
        train_vl_dataset = get_class_instance(
            dataset_config["class_path"], **dataset_config["train"]["init_kwargs"]
        )
        train_vl_dataset.coordinates_adapter = coordinates_adapter
        train_vl_datasets.append(train_vl_dataset)
    train_dataset = ConcatDataset(train_vl_datasets)
    accelerator.print(f"train dataset size: {len(train_dataset)}")
    train_data_loader = DataLoader(
        train_dataset,
        collate_fn=vl_local_model.collate,
        **config["pipeline"]["data_loader_kwargs"],
    )
    train_batch_size = (
        train_data_loader.batch_size
        * accelerator.num_processes
        * accelerator.gradient_accumulation_steps
    )
    accelerator.print(f"train batch size: {train_batch_size}")
    validate_vl_datasets = []
    for dataset_config in config["datasets"]:
        validate_vl_dataset = get_class_instance(
            dataset_config["class_path"], **dataset_config["validate"]["init_kwargs"]
        )
        validate_vl_dataset.coordinates_adapter = coordinates_adapter
        validate_vl_datasets.append(validate_vl_dataset)
    validate_dataset = ConcatDataset(validate_vl_datasets)
    accelerator.print(f"validate dataset size: {len(validate_dataset)}")
    validate_data_loader = DataLoader(
        validate_dataset,
        collate_fn=vl_local_model.collate,
        **config["pipeline"]["data_loader_kwargs"],
    )
    validate_batch_size = validate_data_loader.batch_size * accelerator.num_processes
    accelerator.print(f"validate batch size: {validate_batch_size}")
    optimizer = AdamW(
        vl_local_model.model.parameters(), **config["pipeline"]["adamw_kwargs"]
    )
    num_training_steps = (
        math.ceil(len(train_data_loader) / accelerator.gradient_accumulation_steps)
        * config["pipeline"]["num_epochs"]
    )
    num_warmup_steps = math.ceil(
        num_training_steps * config["pipeline"]["warmup_ratio"]
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps, num_training_steps
    )
    (
        vl_local_model.model,
        train_data_loader,
        validate_data_loader,
        optimizer,
        scheduler,
    ) = accelerator.prepare(
        vl_local_model.model,
        train_data_loader,
        validate_data_loader,
        optimizer,
        scheduler,
    )
    accelerator.print(f"number of epochs: {config['pipeline']['num_epochs']}")
    num_steps_per_epoch = math.ceil(
        len(train_data_loader) / accelerator.gradient_accumulation_steps
    )
    accelerator.print(f"number of steps per epoch: {num_steps_per_epoch}")
    num_steps_in_total = num_steps_per_epoch * config["pipeline"]["num_epochs"]
    accelerator.print(f"number of steps in total: {num_steps_in_total}")
    if config["pipeline"]["source_logging_dir"] is None:
        checkpoint_dir, step = load_checkpoint(accelerator, logging_dir)
    else:
        checkpoint_dir, step = load_checkpoint(
            accelerator,
            logging_dir,
            source_logging_dir=Path(config["pipeline"]["source_logging_dir"]),
        )
    if checkpoint_dir is not None:
        accelerator.print(f"loaded checkpoint from '{checkpoint_dir}'")
    start_epoch = step // num_steps_per_epoch
    epoch_step = step % num_steps_per_epoch
    accelerator.print(
        f"starting from step {step} (step {epoch_step} of epoch {start_epoch})"
    )
    vl_local_model.model.train()
    for epoch in range(start_epoch, config["pipeline"]["num_epochs"]):
        loss_meter = LossMeter()
        progress_bar = tqdm(desc=f"train (epoch {epoch})", total=num_steps_per_epoch)
        if epoch == start_epoch:
            train_data_loader_backup = train_data_loader
            train_data_loader = accelerator.skip_first_batches(
                train_data_loader,
                num_batches=epoch_step * accelerator.gradient_accumulation_steps,
            )
            progress_bar.update(epoch_step)
        accelerator.wait_for_everyone()
        start_time = time.time()
        for vl_local_input in train_data_loader:
            with accelerator.accumulate(vl_local_model.model):
                loss = vl_local_model.forward(vl_local_input)
                loss_meter.update(loss.item(), vl_local_input.labels.shape[0])
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        vl_local_model.model.parameters(),
                        config["pipeline"]["grad_max_norm"],
                    )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                if accelerator.sync_gradients:
                    accelerator.wait_for_everyone()
                    step_time = time.time() - start_time
                    train_loss = [loss_meter.average]
                    train_loss = gather_object(train_loss)
                    train_loss = sum(train_loss) / len(train_loss)
                    lr = scheduler.get_last_lr()[0]
                    step += 1
                    accelerator.log(
                        {
                            "epoch": epoch,
                            "step_time": step_time,
                            "train_loss": train_loss,
                            "lr": lr,
                        },
                        step=step,
                    )
                    loss_meter.reset()
                    progress_bar.update()
                    if (
                        step % config["pipeline"]["num_steps_per_checkpoint"] == 0
                        or step % num_steps_per_epoch == 0
                        or step == num_steps_in_total
                    ):
                        vl_local_model.model.eval()
                        validate_loss = validate(vl_local_model, validate_data_loader)
                        vl_local_model.model.train()
                        accelerator.log({"validate_loss": validate_loss}, step=step)
                        accelerator.print(
                            f"validate_loss: {validate_loss}, step: {step}"
                        )
                        checkpoint_dir, export_dir = save_checkpoint_and_export_model(
                            accelerator, vl_local_model, logging_dir, step
                        )
                        accelerator.print(
                            f"saved checkpoint to '{checkpoint_dir}' and exported model to '{export_dir}'"
                        )
                    accelerator.wait_for_everyone()
                    start_time = time.time()
        if epoch == start_epoch:
            train_data_loader = train_data_loader_backup
        progress_bar.close()
    accelerator.end_training()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline_name", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--dataset_names", nargs="+", type=str, required=True)
    args = parser.parse_args()
    train(args.pipeline_name, args.model_name, args.dataset_names)
