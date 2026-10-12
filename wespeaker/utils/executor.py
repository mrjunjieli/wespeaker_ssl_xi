# Copyright (c) 2026 Junjie Li (junjie98.li@connect.polyu.hk)
#               2021 Hongji Wang (jijijiang77@gmail.com)
#               2022 Chengdong Liang (liangchengdong@mail.nwpu.edu.cn)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import csv
import os

import tableprint as tp

import torch
import torch.distributed as dist
import torchnet as tnt
from wespeaker.dataset.dataset_utils import apply_cmvn, spec_aug


_UNC_LOG_FIELDS = [
    'epoch', 'iter', 'global_step', 'lr', 'margin', 'loss', 'acc',
    'mult_mean', 'mult_min', 'mult_max', 'mult_std',
    'mult_p10', 'mult_p50', 'mult_p90', 'eff_scale_mean', 'uncertainty_weight'
]


def _is_rank0():
    return (not dist.is_available()) or (not dist.is_initialized()) \
        or dist.get_rank() == 0


# Lazily-created, per-exp_dir TensorBoard writer (rank 0 only). Kept as a
# module-level cache so run_epoch can log without changing train.py's call
# signature; SummaryWriter appends across epochs to the same logdir.
_TB_WRITERS = {}


def _get_tb_writer(configs):
    if not _is_rank0():
        return None
    logdir = os.path.join(configs['exp_dir'], 'tensorboard')
    writer = _TB_WRITERS.get(logdir)
    if writer is None:
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError:
            _TB_WRITERS[logdir] = False
            return None
        writer = SummaryWriter(log_dir=logdir)
        _TB_WRITERS[logdir] = writer
    return writer or None


def _log_uncertainty_scale(configs, row):
    """Append one row of uncertainty-multiplier stats to a CSV in exp_dir.

    Only rank 0 writes. The CSV can be turned into a TensorBoard-style curve
    with local/plot_uncertainty_scale.py.
    """
    if not _is_rank0():
        return
    csv_path = os.path.join(configs['exp_dir'], 'uncertainty_scale_log.csv')
    write_header = not os.path.exists(csv_path)
    with open(csv_path, 'a', newline='') as fout:
        writer = csv.DictWriter(fout, fieldnames=_UNC_LOG_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def run_epoch(dataloader, epoch_iter, model, criterion, optimizer, scheduler,
              margin_scheduler, uncertainty_scheduler, epoch, logger, scaler,
              device, configs):
    model.train()
    # By default use average pooling
    loss_meter = tnt.meter.AverageValueMeter()
    acc_meter = tnt.meter.ClassErrorMeter(accuracy=True)

    frontend_type = configs['dataset_args'].get('frontend', 'fbank')
    # Per-epoch pool of uncertainty_multiplier values, used to emit one
    # histogram per epoch (x = multiplier/scale value, y = count) so epochs
    # can be compared in TensorBoard's HISTOGRAMS/DISTRIBUTIONS tab.
    epoch_mult_values = []
    for i, batch in enumerate(dataloader):
        cur_iter = (epoch - 1) * epoch_iter + i
        scheduler.step(cur_iter)
        margin_scheduler.step(cur_iter)
        if uncertainty_scheduler is not None:
            uncertainty_scheduler.step(cur_iter)

        utts = batch['key']
        targets = batch['label']
        targets = targets.long().to(device)  # (B)
        if frontend_type == 'fbank':
            features = batch['feat']  # (B,T,F)
            features = features.float().to(device)
        else:  # 's3prl', 'tfmel'
            wavs = batch['wav']  # (B,1,W)
            wavs = wavs.squeeze(1).float().to(device)  # (B,W)
            wavs_len = torch.LongTensor([wavs.shape[1]]).repeat(
                wavs.shape[0]).to(device)  # (B)
            with torch.cuda.amp.autocast(enabled=configs['enable_amp']):
                features, _ = model.module.frontend(wavs, wavs_len)

        with torch.cuda.amp.autocast(enabled=configs['enable_amp']):
            # apply cmvn
            if configs['dataset_args'].get('cmvn', True):
                features = apply_cmvn(
                    features, **configs['dataset_args'].get('cmvn_args', {}))
            # spec augmentation
            if configs['dataset_args'].get('spec_aug', False):
                features = spec_aug(features,
                                    **configs['dataset_args']['spec_aug_args'])

            outputs = model(features)  # (embed_a,embed_b) in most cases
            embeds = outputs[-1] if isinstance(outputs, tuple) else outputs
            if getattr(model.module.projection,
                       'requires_uncertainty', False):
                if not isinstance(outputs, tuple) or len(outputs) != 3:
                    raise ValueError(
                        "Uncertainty projection requires model outputs "
                        "(frame_features, variance, embedding)")
                outputs = model.module.projection(embeds, outputs[1], targets)
            else:
                outputs = model.module.projection(embeds, targets)
            if isinstance(outputs, tuple):
                outputs, loss = outputs
            else:
                loss = criterion(outputs, targets)

        # loss, acc
        loss_meter.add(loss.item())
        acc_meter.add(outputs.cpu().detach().numpy(), targets.cpu().numpy())

        # updata the model
        optimizer.zero_grad()
        # scaler does nothing here if enable_amp=False
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # log
        if (i + 1) % configs['log_batch_interval'] == 0:
            logger.info(
                tp.row((epoch, i + 1, scheduler.get_lr(),
                        margin_scheduler.get_margin()) +
                       (loss_meter.value()[0], acc_meter.value()[0]),
                       width=10,
                       style='grid'))
            writer = _get_tb_writer(configs)
            if writer is not None:
                writer.add_scalar('train/loss', loss_meter.value()[0],
                                  cur_iter)
                writer.add_scalar('train/acc', acc_meter.value()[0], cur_iter)
                writer.add_scalar('train/lr', scheduler.get_lr(), cur_iter)
                writer.add_scalar('train/margin',
                                  margin_scheduler.get_margin(), cur_iter)

            unc_stats = getattr(model.module.projection,
                                'last_uncertainty_stats', None)
            if unc_stats is not None:
                _log_uncertainty_scale(
                    configs, {
                        'epoch': epoch,
                        'iter': i + 1,
                        'global_step': cur_iter,
                        'lr': scheduler.get_lr(),
                        'margin': margin_scheduler.get_margin(),
                        'loss': loss_meter.value()[0],
                        'acc': acc_meter.value()[0],
                        **unc_stats,
                    })
                if writer is not None:
                    for k, v in unc_stats.items():
                        writer.add_scalar('uncertainty/' + k, v, cur_iter)
                    writer.flush()
                # Collect raw multiplier values for this epoch's histogram
                # (rank 0 only, D2H copy happens here at log cadence).
                vals = getattr(model.module.projection,
                               'last_uncertainty_values', None)
                if _is_rank0() and vals is not None:
                    epoch_mult_values.append(vals.detach().cpu())

        if (i + 1) == epoch_iter:
            break

    # One histogram per epoch: pool all collected multiplier values and tag it
    # with global_step=epoch so TensorBoard's HISTOGRAMS tab overlays epochs
    # (x = uncertainty_multiplier value, y = count).
    if epoch_mult_values:
        writer = _get_tb_writer(configs)
        if writer is not None:
            all_vals = torch.cat(epoch_mult_values)
            writer.add_histogram('uncertainty/mult_hist', all_vals, epoch)
            writer.flush()

    logger.info(
        tp.row(
            (epoch, i + 1, scheduler.get_lr(), margin_scheduler.get_margin()) +
            (loss_meter.value()[0], acc_meter.value()[0]),
            width=10,
            style='grid'))
