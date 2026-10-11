# DeepANC

DeepANC 論文の手法を Python で実装した実験用リポジトリです。Python モジュールは `src/` にあり、実験は Jupyter Notebook から実行します。

## ディレクトリ構成

```text
.
├── 1.Synthesize_data/              # 合成学習音声と室内インパルス応答の生成
├── 2.Train_Network/                # ネットワークの学習
├── 3.Noise_Cancellation_Code/      # ノイズキャンセルの推論・評価
├── src/
│   ├── common/                     # モデル、データローダー、信号処理
│   ├── synthesize/                 # 合成音声生成
│   ├── train/                      # 学習処理
│   └── inference/                  # 推論処理
├── pyproject.toml                  # src レイアウトの Python パッケージ設定
└── requirements_for_titanx.txt     # 実験環境の依存パッケージ
```

## GitHub に含まれないファイル

`.gitignore` の設定により、次のファイルやディレクトリは通常 GitHub にアップロードされません。

- `documents/`（論文 PDF など）
- `Dataset/`、`datasets/`、`data/` 内のデータ
- `*.wav`、`*.mp3`、`*.flac` の音声ファイル
- `*.mat` の室内音響応答ファイル
- `models/` は名前では除外されませんが、`*.pth`、`*.pt`、`*.ckpt` の学習済み重みは除外されます
- `checkpoints/`、`outputs/`、`results/`、`runs/`、`tensorboard/` などの実験出力
- 仮想環境、Python キャッシュ、Jupyter checkpoint、ログ、`.env` など

そのため、クローン直後にはデータセット、RIR、学習済み重み、論文 PDF が存在しない場合があります。論文 PDF は別途入手して `documents/` に置いてください。実録騒音を使う場合は、利用許可のある音源を別途準備してください。音源・重みなどを Git に追加する場合は、`.gitignore` の対象であることを確認してください。

### 参考論文と入手元

`documents/` に置く論文の掲載元・ランディングページは以下のとおりです。本文や PDF の入手可否は出版社・会議の公開条件に従ってください。

- Ke Tan and DeLiang Wang, “Complex Spectral Mapping with a Convolutional Recurrent Network for Monaural Speech Enhancement,” ICASSP 2019. [IEEE Xplore](https://ieeexplore.ieee.org/document/8682834)
- Hao Zhang and DeLiang Wang, “Deep ANC: A deep learning approach to active noise control,” *Neural Networks*, 141, 1–10 (2021). [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0893608021001258)
- Fei Gao et al., “Efficient Sequence Learning with Group Recurrent Networks,” NAACL-HLT 2018. [ACL Anthology](https://aclanthology.org/N18-1073/)
- Hao Zhang and DeLiang Wang, “Deep MCANC: A Deep Learning Approach to Multi-Channel Active Noise Control,” *Neural Networks*, 158, 318–327 (2023). [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0893608022004701)

## 実行環境の準備

Python 環境を有効化したうえで、リポジトリのルートで次を実行します。

```bash
python -m pip install -r requirements_for_titanx.txt
python -m pip install -e .
```

`pyproject.toml` は `src/` 配下を `synthesize`、`train`、`common`、`inference` としてインストールする設定です。依存パッケージのバージョンは `requirements_for_titanx.txt` を参照してください。記載された PyTorch / torchaudio の組み合わせや GPU 利用可否は環境に合わせて調整が必要な場合があります。

Jupyter が起動できない場合は、環境に `ipykernel` と Notebook 実行環境を用意してください。

## 合成音声による動作確認

このリポジトリで確認しやすい学習手順は、`1.Synthesize_data/make_synthesized_data.ipynb` で生成する合成バンド制限ノイズを使うものです。ノートブックはサンプリング周波数 16 kHz、長さ 10 秒、周波数帯域 20–7980 Hz の音声を 20,000 個生成し、`Dataset/train20000/` に保存します。

1. `1.Synthesize_data/make_synthesized_data.ipynb` を開き、リポジトリルートをカレントディレクトリとして実行します。`PROJECT_ROOT = Path.cwd().parent` としているため、このノートブックのカレントディレクトリは `1.Synthesize_data/` にしてください。生成には十分なディスク容量が必要です。
2. 必要に応じて `1.Synthesize_data/make_rir.ipynb` を実行します。シミュレーションした室内応答を `rirs/rt60_...s/` に `.mat` ファイルとして生成します。
3. 生成した音声を `Dataset/train/` と `Dataset/validate/` に分けて配置します。`2.Train_Network/Train_Network.ipynb` はこの2ディレクトリを読み込みます。音声生成ノートブックの保存先は `Dataset/train20000/` なので、学習前にファイルを学習用・検証用へ分割してください。`MyDataset` は指定フォルダー直下のファイルを読み込むため、サブフォルダーに分けず WAV ファイルを直接配置します。
4. `2.Train_Network/Train_Network.ipynb` を開き、カレントディレクトリを `2.Train_Network/` にして実行します。既定設定では `rirs/` の応答を使い、30 エポック学習して、検証 NMSE が改善したモデルを `models/synth_delay_1.pth` に保存します。`Dataset/`、`rirs/`、`models/` はリポジトリルート直下に配置します。

`make_rir.ipynb` は `rirs/` 以下の `.mat` ファイルを、学習ノートブックは `Dataset/train`、`Dataset/validate` の WAV と `models/` 以下の重みをそれぞれ必要とします。これらは `.gitignore` の対象を含むため、ノートブックを実行する前にローカルで作成してください。

## 論文の実録騒音条件と、このリポジトリの実装

論文では実録騒音を用いた学習・評価が記述されています。一方、このリポジトリで動作確認しやすい `make_synthesized_data.ipynb` と `Train_Network.ipynb` の学習フローは、合成バンド制限ノイズを使います。学習コードは WAV を読み込み、シミュレーションした一次経路・二次経路をランダムに適用して学習データを構成します。したがって、この合成音声による学習は、論文の実録騒音を使った学習条件そのものではなく、実装の動作確認用の手順です。

`3.Noise_Cancellation_Code/Noise_Cancellation.ipynb` には実録音源の読み込み処理があります。評価に使う WAV を `Dataset/disturbance/` に置き、ノートブックの `sound_name` をそのファイル名に変更してください。ローダーは音源を読み込み、16 kHz にリサンプリングします。ノートブックの例の `1_from_1733_to_2243_Hz.wav` はファイル名の例にすぎず、リポジトリには音源が含まれません。

実録騒音を用いた学習を再現するには、論文で指定された実録音源・学習データの分割・収録条件を確認し、学習用音声を `Dataset/train/` と `Dataset/validate/` に用意してください。現在の学習コードはこれらのフォルダー内の WAV をデータとして読み込みますが、合成音声生成ノートブックが自動で実録音源を用意したり、データを分割したりはしません。論文 PDF は `documents/` 以下で参照できる想定ですが、このディレクトリは GitHub には含まれません。

## ノートブックの実行順

```text
1.Synthesize_data/make_synthesized_data.ipynb   合成学習音声を生成
1.Synthesize_data/make_rir.ipynb                シミュレーション RIR を生成
                 ↓ Dataset/train と Dataset/validate を用意
2.Train_Network/Train_Network.ipynb              モデルを学習
                 ↓ models/synth_delay_1.pth
3.Noise_Cancellation_Code/Noise_Cancellation.ipynb  実録音源でノイズキャンセルを評価
```

ノートブック内で使う相対パスを合わせるため、各ノートブックはそれぞれ記載のサブディレクトリをカレントディレクトリとして実行してください。
