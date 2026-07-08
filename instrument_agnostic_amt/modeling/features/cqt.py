import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.signal
import math


class RecursiveCQT(nn.Module):
    """
    再帰的ダウンサンプリングを用いた高速・省メモリなCQT実装。
    """

    def __init__(
        self,
        sr=22050,
        hop_length=512,
        fmin=32.7,
        n_bins=84,
        bins_per_octave=12,
        window="hann",
        resampling_quality=64,
        filter_scale=1.0,
    ):
        """
        モジュールの初期化と、CQTカーネルおよびダウンサンプラーの事前計算を行います。
        """
        super().__init__()
        # 基本パラメータ
        self.sr = sr
        self.hop_length = hop_length
        self.fmin = fmin
        self.n_bins = n_bins
        self.bins_per_octave = bins_per_octave
        self.window = window

        if self.sr <= 0:
            raise ValueError("sr must be positive")
        if self.hop_length <= 0:
            raise ValueError("hop_length must be positive")
        if self.hop_length & (self.hop_length - 1) != 0:
            raise ValueError("hop_length must be a power of two")
        if self.fmin <= 0:
            raise ValueError("fmin must be positive")
        if self.n_bins <= 0:
            raise ValueError("n_bins must be positive")
        if self.bins_per_octave <= 0:
            raise ValueError("bins_per_octave must be positive")

        # CQT計算用の定数を設定
        # Q値: CQTの周波数解像度を決定する定数
        self.q = filter_scale / (2.0 ** (1.0 / bins_per_octave) - 1.0)
        # 全てのCQTビンの中心周波数を計算
        all_freqs = self.fmin * 2.0 ** (np.arange(self.n_bins) / self.bins_per_octave)

        # 部分オクターブを新しいステージにしないよう、最上段が端数を吸収する。
        # 例: 312 bins / 36 bins_per_octave -> [0:36), ..., [216:252), [252:312)
        if self.n_bins <= self.bins_per_octave:
            self.stage_ranges = [(0, self.n_bins)]
        else:
            remainder = self.n_bins % self.bins_per_octave
            top_stage_bins = (
                self.bins_per_octave
                if remainder == 0
                else self.bins_per_octave + remainder
            )
            lower_bins = self.n_bins - top_stage_bins
            self.stage_ranges = [
                (start, start + self.bins_per_octave)
                for start in range(0, lower_bins, self.bins_per_octave)
            ]
            self.stage_ranges.append((lower_bins, self.n_bins))
        self.n_octaves = len(self.stage_ranges)

        if self.hop_length < (2 ** (self.n_octaves - 1)):
            raise ValueError(
                "hop_length is too small for the number of recursive stages"
            )

        self.resamplers = nn.ModuleList()
        self.fft_sizes = []
        # 現在の処理ステージでのサンプリングレート
        current_sr = sr

        # ステージごとのループ処理
        # 高い周波数帯から低い周波数帯へ順に処理します。
        # iはダウンサンプリングのステージを表します (i=0はダウンサンプリングなし)。
        for i, (octave_start_bin, octave_end_bin) in enumerate(
            reversed(self.stage_ranges)
        ):
            # ビンが存在しない場合はスキップ
            if octave_start_bin >= octave_end_bin:
                continue

            # このオクターブ内の周波数を取得
            octave_freqs = all_freqs[octave_start_bin:octave_end_bin]
            if float(octave_freqs[-1]) > float(current_sr) / 2.0:
                raise ValueError(
                    "Requested CQT bins exceed the Nyquist frequency for this stage"
                )

            # FFTサイズを決定: このオクターブの最低周波数（最も長いカーネルが必要）を
            # カバーできる最小の2のべき乗を計算します。
            min_freq_in_octave = octave_freqs[0]
            fft_size = 2 ** int(
                np.ceil(np.log2(self.q * current_sr / min_freq_in_octave))
            )
            self.fft_sizes.append(fft_size)

            # STFTからCQTへの変換カーネルを事前計算し、バッファに登録
            kernel = self._create_cqt_kernel(octave_freqs, current_sr, fft_size)
            self.register_buffer(f"cqt_kernel_{i}", kernel, persistent=False)

            # STFT計算用の窓関数も同様にバッファに登録
            win = scipy.signal.get_window(self.window, fft_size)
            self.register_buffer(
                f"window_{i}", torch.from_numpy(win).float(), persistent=False
            )

            # 次のステージ（より低いオクターブ）のためにダウンサンプラーを構築
            if i < self.n_octaves - 1:
                # FIRフィルタを設計し、アンチエイリアシングフィルタとして使用
                kaiser_beta = 5.0
                fir_coeffs = scipy.signal.firwin(
                    resampling_quality + 1,
                    current_sr / 4.0,
                    fs=current_sr,
                    window=("kaiser", kaiser_beta),
                )

                # Conv1dでダウンサンプリングを実装
                padding = resampling_quality // 2
                resampler = nn.Conv1d(
                    1,
                    1,
                    kernel_size=resampling_quality + 1,
                    stride=2,
                    bias=False,
                    padding=padding,
                )

                # フィルタ係数を重みとして設定し、学習しないように固定
                resampler.weight.data = (
                    torch.from_numpy(fir_coeffs).float().view(1, 1, -1)
                )
                resampler.requires_grad_(False)
                self.resamplers.append(resampler)

            # 次のステージのためにサンプリングレートを半分にする
            current_sr /= 2.0

    def _create_cqt_kernel(self, freqs, fs, n_fft):
        """
        指定されたオクターブのSTFT->CQT変換カーネルを作成するヘルパー関数。
        """
        n_freqs = len(freqs)
        # このオクターブの全周波数に対するカーネルを格納するゼロ行列
        kernel = torch.zeros((n_freqs, n_fft // 2 + 1), dtype=torch.complex64)

        # 各周波数についてループ処理
        for k, f in enumerate(freqs):
            # 周波数に応じた長さの複素カーネルを時間領域で作成
            length = self.q * fs / f
            if length > n_fft:
                length = n_fft
            win = torch.from_numpy(
                scipy.signal.get_window(self.window, int(np.ceil(length)))
            ).float()
            time_idx = torch.arange(len(win), dtype=torch.float32)
            sinusoid = torch.exp(2j * np.pi * f * time_idx / fs)
            time_kernel = win * sinusoid

            # L1ノルムで正規化し、エネルギーを揃える
            norm = torch.sum(torch.abs(time_kernel))
            if norm > 1e-8:
                time_kernel /= norm

            # FFTサイズに合わせてカーネルを中央に配置
            padded_kernel = torch.zeros(n_fft, dtype=torch.complex64)
            start = (n_fft - len(time_kernel)) // 2
            padded_kernel[start : start + len(time_kernel)] = time_kernel

            # FFTで周波数領域のカーネルに変換
            spec = torch.fft.fft(padded_kernel)
            # 実数信号のFFT（rfft）の出力サイズに合わせる
            spec = spec[..., : n_fft // 2 + 1]
            kernel[k, :] = spec

        kernel.requires_grad = False
        return kernel

    def forward(self, x, return_complex=False):
        """
        モジュールの順伝播。入力音声波形からCQTスペクトログラムを計算します。
        """
        # 入力がバッチ次元を持たない場合、追加する
        if x.ndim == 1:
            x = x.unsqueeze(0)

        # 基準となるSTFTが出力すべき目標フレーム数を計算
        # これにより、最終的な出力長がSTFTと一致する
        target_frames = math.ceil(x.shape[-1] / self.hop_length)

        cqt_outputs = []
        current_x = x

        # __init__で構築した順（高周波オクターブから）に処理
        for i in range(len(self.fft_sizes)):
            # このステージで使用するパラメータを取得
            fft_size = self.fft_sizes[i]
            window = getattr(self, f"window_{i}")
            cqt_kernel = getattr(self, f"cqt_kernel_{i}")

            # ダウンサンプリングに応じてホップ長を調整
            current_hop = self.hop_length // (2**i)

            # STFTを計算
            stft_out = torch.stft(
                current_x,
                n_fft=fft_size,
                hop_length=current_hop,
                win_length=fft_size,
                window=window.to(current_x.device),
                center=True,
                pad_mode="reflect",
                return_complex=True,
            )

            # STFTの結果とCQTカーネルの内積を計算してCQTに変換
            cqt_octave = torch.matmul(
                stft_out.transpose(-1, -2), cqt_kernel.T.to(current_x.device)
            )
            cqt_octave = cqt_octave.transpose(-1, -2)

            # 各オクターブの出力を目標フレーム数にパディングまたはトリミング
            current_frames = cqt_octave.shape[-1]
            if current_frames < target_frames:
                # フレーム数が足りない場合はゼロでパディング
                padding_needed = target_frames - current_frames
                cqt_octave = F.pad(cqt_octave, (0, padding_needed))
            elif current_frames > target_frames:
                # フレーム数が多すぎる場合は切り捨てる
                cqt_octave = cqt_octave[..., :target_frames]

            cqt_outputs.append(cqt_octave)

            # 次のステージのために信号をダウンサンプリング
            if i < len(self.resamplers):
                current_x = self.resamplers[i](current_x.unsqueeze(1)).squeeze(1)

        # 結果のリストは[高オクターブ, ..., 低オクターブ]の順なので、逆順にする
        cqt_outputs.reverse()
        # 周波数ビン次元（dim=1）で全オクターブの結果を結合
        cqt_spectrogram = torch.cat(cqt_outputs, dim=1)

        return cqt_spectrogram if return_complex else torch.abs(cqt_spectrogram)
