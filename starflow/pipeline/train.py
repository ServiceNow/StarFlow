from accelerate.utils import gather_object, tqdm
from accelerate.utils.modeling import load_checkpoint_in_model
from omegaconf import OmegaConf
from torch import inference_mode
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup
from starflow.model.base import VLModel
from starflow.pipeline.utils import (
    get_accelerator,
    get_config,
    get_logging_dir,
    LossMeter,
    load_checkpoint,
    save_checkpoint,
)
from starflow.utils import get_vl_object
import logging
import math
import time
import warnings


@inference_mode()
def validate(vl_model: VLModel, validate_data_loader: DataLoader):
    loss_meter = LossMeter()
    progress_bar = tqdm(desc="Validate", total=len(validate_data_loader))
    for vl_input in validate_data_loader:
        loss = vl_model(vl_input)
        loss_meter.update(loss.item(), vl_input.labels.shape[0])
        progress_bar.update()
    progress_bar.close()
    validate_loss = [loss_meter.average]
    validate_loss = gather_object(validate_loss)
    validate_loss = sum(validate_loss) / len(validate_loss)
    return validate_loss


def train():
    config = get_config()
    logging_dir = get_logging_dir(config)
    vl_model = get_vl_object(
        config.model.class_path, **OmegaConf.to_container(config.model.init_kwargs)
    )
    accelerator = get_accelerator(config, logging_dir, vl_model)
    accelerator.print(f"Logging dir: {logging_dir}")
    accelerator.print(f"Dataset: {config.dataset.name}")
    accelerator.print(f"Model: {config.model.name}")
    accelerator.print(f"Pipeline: {config.pipeline.name}")
    if config.pipeline.model_checkpoint_file is not None:
        load_checkpoint_in_model(vl_model, config.pipeline.model_checkpoint_file)
        accelerator.print(
            f"Loaded model checkpoint from {config.pipeline.model_checkpoint_file}"
        )
    accelerator.print(f"Num processes: {accelerator.num_processes}")
    accelerator.print(
        f"Gradient accumulation steps: {accelerator.gradient_accumulation_steps}"
    )
    train_vl_dataset = get_vl_object(
        config.dataset.class_path,
        **OmegaConf.to_container(config.dataset.train.init_kwargs),
    )
    accelerator.print(f"Train dataset size: {len(train_vl_dataset)}")
    train_data_loader = DataLoader(
        train_vl_dataset,
        collate_fn=vl_model.collate,
        **OmegaConf.to_container(config.pipeline.data_loader_kwargs),
    )
    train_batch_size_per_process = train_data_loader.batch_size
    accelerator.print(f"Train batch size per process: {train_batch_size_per_process}")
    train_batch_size_in_total = (
        train_batch_size_per_process
        * accelerator.num_processes
        * accelerator.gradient_accumulation_steps
    )
    accelerator.print(f"Train batch size in total: {train_batch_size_in_total}")
    validate_vl_dataset = get_vl_object(
        config.dataset.class_path,
        **OmegaConf.to_container(config.dataset.validate.init_kwargs),
    )
    accelerator.print(f"Validate dataset size: {len(validate_vl_dataset)}")
    validate_data_loader = DataLoader(
        validate_vl_dataset,
        collate_fn=vl_model.collate,
        **OmegaConf.to_container(config.pipeline.data_loader_kwargs),
    )
    validate_batch_size_per_process = validate_data_loader.batch_size
    accelerator.print(
        f"Validate batch size per process: {validate_batch_size_per_process}"
    )
    validate_batch_size_in_total = (
        validate_batch_size_per_process * accelerator.num_processes
    )
    accelerator.print(f"Validate batch size in total: {validate_batch_size_in_total}")
    optimizer = AdamW(
        vl_model.parameters(), **OmegaConf.to_container(config.pipeline.adamw_kwargs)
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.pipeline.num_warmup_steps,
        num_training_steps=math.ceil(
            len(train_data_loader) / accelerator.gradient_accumulation_steps
        )
        * config.pipeline.num_epochs,
    )
    vl_model, train_data_loader, validate_data_loader, optimizer, scheduler = (
        accelerator.prepare(
            vl_model, train_data_loader, validate_data_loader, optimizer, scheduler
        )
    )
    vl_model.train()
    accelerator.print(f"Num epochs: {config.pipeline.num_epochs}")
    num_steps_per_epoch = math.ceil(
        len(train_data_loader) / accelerator.gradient_accumulation_steps
    )
    accelerator.print(f"Num steps per epoch: {num_steps_per_epoch}")
    num_steps_in_total = num_steps_per_epoch * config.pipeline.num_epochs
    accelerator.print(f"Num steps in total: {num_steps_in_total}")
    checkpoint_dir, overall_step = load_checkpoint(config, accelerator)
    start_step = overall_step % num_steps_per_epoch
    start_epoch = overall_step // num_steps_per_epoch
    if checkpoint_dir is not None:
        accelerator.print(f"Loaded checkpoint from {checkpoint_dir}")
    accelerator.print(
        f"Started from overall step {overall_step} (step {start_step} of epoch {start_epoch})"
    )
    for epoch in range(start_epoch, config.pipeline.num_epochs):
        loss_meter = LossMeter()
        progress_bar = tqdm(desc=f"Epoch {epoch}", total=num_steps_per_epoch)
        if epoch == start_epoch:
            train_data_loader_backup = train_data_loader
            train_data_loader = accelerator.skip_first_batches(
                train_data_loader, start_step * accelerator.gradient_accumulation_steps
            )
            progress_bar.update(start_step)
        else:
            train_data_loader = train_data_loader_backup
        accelerator.wait_for_everyone()
        start_time = time.time()
        for vl_input in train_data_loader:
            with accelerator.accumulate(vl_model):
                loss = vl_model(vl_input)
                loss_meter.update(loss.item(), vl_input.labels.shape[0])
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        vl_model.parameters(), config.pipeline.max_grad_norm
                    )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                if accelerator.sync_gradients:
                    overall_step += 1
                    train_loss = [loss_meter.average]
                    train_loss = gather_object(train_loss)
                    train_loss = sum(train_loss) / len(train_loss)
                    lr = scheduler.get_last_lr()[0]
                    accelerator.wait_for_everyone()
                    step_time = time.time() - start_time
                    accelerator.log(
                        {
                            "train_loss": train_loss,
                            "lr": lr,
                            "step_time": step_time,
                            "epoch": epoch,
                            "overall_step": overall_step,
                        },
                        overall_step,
                    )
                    loss_meter.reset()
                    progress_bar.update()
                    if (
                        overall_step % config.pipeline.num_steps_per_checkpoint == 0
                        or overall_step % num_steps_per_epoch == 0
                        or overall_step == num_steps_in_total
                    ):
                        vl_model.eval()
                        validate_loss = validate(vl_model, validate_data_loader)
                        vl_model.train()
                        accelerator.print(
                            f"validate_loss: {validate_loss}, epoch: {epoch}, overall_step: {overall_step}"
                        )
                        accelerator.log(
                            {
                                "validate_loss": validate_loss,
                                "epoch": epoch,
                                "overall_step": overall_step,
                            },
                            overall_step,
                        )
                        checkpoint_dir = save_checkpoint(accelerator, overall_step)
                        accelerator.print(f"Saved checkpoint to {checkpoint_dir}")
                    accelerator.wait_for_everyone()
                    start_time = time.time()
        progress_bar.close()
    accelerator.end_training()


if __name__ == "__main__":
    logging.disable(logging.WARNING)
    warnings.filterwarnings("ignore")
    train()
