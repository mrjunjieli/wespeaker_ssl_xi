import numpy as np
import torch
import torch.nn.functional as F
from transformers.modeling_outputs import BaseModelOutput


def _get_model_device_dtype(encoder):
    """获取模型的设备和数据类型"""
    device = next(encoder.parameters()).device
    dtype = next(encoder.parameters()).dtype
    return device, dtype


def _extract_fbank_gpu(self, aud_inputs):
    """GPU 版 fbank，复现 SeamlessM4TFeatureExtractor 的 mel 计算。

    与 CPU 版本（transformers 的 feature_extractor）保持相同参数：
      frame_length=400, hop_length=160, fft_length=512, power=2.0,
      center=False, preemphasis=0.97, remove_dc_offset=True,
      kaldi mel filters (20~8000Hz, 80 bins), log-mel, per-bin 归一化,
      stride=2 reshape。

    Args:
        self: HuggingfaceFrontend_w2vbert 实例（含 feature_extractor）
        aud_inputs: (B, T) float tensor on GPU

    Returns:
        input_features: (B, T//2, 160)
    """
    device = aud_inputs.device

    # mel_filters / window 从 feature_extractor 缓存到 GPU（首次或设备变化时）
    if not hasattr(self, '_mel_filters_gpu') or self._mel_filters_gpu.device != device:
        self._mel_filters_gpu = torch.from_numpy(
            self.feature_extractor.mel_filters.astype(np.float32)).to(device)
        self._window_gpu = torch.from_numpy(
            self.feature_extractor.window.astype(np.float32)).to(device)

    mel_filters = self._mel_filters_gpu  # (257, 80)
    window = self._window_gpu            # (400,)

    frame_length = 400
    hop_length = 160
    fft_length = 512
    mel_floor = 1.192092955078125e-07  # 2^-23
    preemphasis = 0.97
    stride = 2

    # fbank 是确定性变换，无需梯度；且强制 float32 避免 AMP 把 matmul 降到 fp16
    with torch.no_grad(), torch.amp.autocast('cuda', enabled=False):
        # 1. 16-bit 缩放（Kaldi compliance）
        x = aud_inputs.float() * (2 ** 15)  # (B, T)

        # 2. 分帧（center=False）
        frames = x.unfold(1, frame_length, hop_length)  # (B, num_frames, 400)

        # 3. remove_dc_offset
        frames = frames - frames.mean(dim=-1, keepdim=True)

        # 4. pre-emphasis
        pre = frames.clone()
        pre[..., 1:] = pre[..., 1:] - preemphasis * frames[..., :-1]
        pre[..., 0] = pre[..., 0] * (1.0 - preemphasis)

        # 5. window
        pre = pre * window

        # 6. FFT（补零到 fft_length）
        pre = F.pad(pre, (0, fft_length - frame_length))  # (B, num_frames, 512)
        spec = torch.fft.rfft(pre, n=fft_length)  # (B, num_frames, 257)

        # 7. power spectrogram
        power = spec.abs() ** 2  # (B, num_frames, 257)

        # 8. mel filter bank
        mel = power @ mel_filters  # (B, num_frames, 80)

        # 9. log-mel（带 floor）
        log_mel = torch.log(torch.clamp(mel, min=mel_floor))

        # 10. per-mel-bin 归一化（ddof=1）
        mean = log_mel.mean(dim=1, keepdim=True)
        var = log_mel.var(dim=1, keepdim=True, unbiased=True)
        log_mel = (log_mel - mean) / torch.sqrt(var + 1e-7)

        # 11. stride reshape: (B, num_frames // 2, 160)
        num_frames = log_mel.shape[1]
        num_frames = num_frames - (num_frames % stride)
        log_mel = log_mel[:, :num_frames, :]
        log_mel = log_mel.reshape(log_mel.shape[0], num_frames // stride, -1)

    return log_mel


def forward_w2v_bert(self, aud_inputs):
    """
    W2v-BERT 模型的专用 forward 逻辑

    Args:
        self: HuggingfaceFrontend_w2vbert 实例
        aud_inputs: 音频输入张量，形状为 (B, T)

    Returns:
        BaseModelOutput: 包含 last_hidden_state 和 hidden_states 的输出
    """
    # 获取模型参数所在的设备和数据类型
    device, dtype = _get_model_device_dtype(self.upstream)

    # 在 GPU 上计算 fbank（替代 CPU 上的 feature_extractor）
    input_features = _extract_fbank_gpu(self, aud_inputs).to(device).to(dtype)

    # 特征投影
    x = self.upstream.feature_projection(input_features)[0]

    # 编码器层
    hidden_states = [x]
    for layer in self.upstream.encoder.layers:
        x = layer(x)[0]
        hidden_states.append(x)

    # 返回标准格式
    return BaseModelOutput(
        last_hidden_state=x,
        hidden_states=tuple(hidden_states),
    )
