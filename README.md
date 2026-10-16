# DeepANC

DeepANC の論文「Deep ANC: A deep learning approach to active noise control」の手法を実装した研究・実験用 Python プロジェクトです。合成雑音の生成、室内インパルス応答（RIR）の生成、ネットワーク学習、ANC 推論のサンプルを Jupyter Notebook で実行できます。

このリポジトリは実験コードを提供します。音声データ、RIR、学習済みモデルは含まれておらず、ノートブックもそれらを自動ダウンロードしません。

## ディレクトリ構成

```text
.
├── notebooks/
│   ├── 1.Synthesize_data/
│   │   ├── make_synthesized_data.ipynb  # 学習・検証・テスト用の合成雑音を作成
│   │   └── make_rir.ipynb               # シミュレーション RIR を作成
│   ├── 2.Train_Network/
│   │   └── Train_Network.ipynb          # モデルを学習
│   └── 3.Noise_Cancellation_Code/
│       └── Noise_Cancellation.ipynb     # 学習済みモデルで ANC を実行・評価
├── src/
│   └── deepanc/
│       ├── common/                      # ネットワーク、データローダー、信号処理
│       ├── synthesize/                  # 合成信号の生成
│       ├── train/                       # 学習処理
│       └── inference/                   # 推論・実録音源の読み込み
├── tests/
│   └── test_import.py
├── pyproject.toml                       # パッケージと依存関係の設定
└── .gitignore
```

## 必要環境とインストール

`pyproject.toml` は Python 3.9 以上を指定しています。ノートブックを実行するには、リポジトリのルートで仮想環境を作成し、`normal` extra とともにパッケージをインストールします。

```bash
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows PowerShell の場合: .venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -e ".[normal]"
```

GPU を利用する場合は、利用する OS・CUDA 環境に合った PyTorch と torchaudio を選んでください。`pyproject.toml` の `titanx` extra は特定環境向けの依存バージョンを記載しています。環境に合うことを確認してから利用してください。

## データと生成物

ノートブックは `pyprojroot.here()` でプロジェクトルートを見つけ、以下のディレクトリを読み書きします。必要なディレクトリは、生成ノートブックまたは学習ノートブックが作成するものもあります。

```text
Dataset/
├── train/       # 学習用 WAV（直下に配置）
├── validate/    # 検証用 WAV（直下に配置）
└── test/        # 推論・評価用 WAV（直下に配置）
rirs/
└── rt60_0.2s/   # Primary_path.mat、Secondary_path.mat など
models/          # 学習済み重み
figs/rirs/       # RIR ノートブックの図
```

`.gitignore` は `NOISEX92/`、`*.wav`、`*.mat`、`*.pth` などを除外します（`Dataset/` ディレクトリ自体を指定した除外ルールはありませんが、配下の WAV は Git の対象外です）。これらのデータや生成物はローカルで用意してください。`MyDataset` は指定したフォルダー直下のファイルを読み込むため、WAV を追加のサブフォルダーに入れず `Dataset/train/` または `Dataset/validate/` に配置します。

## ノートブックによる基本実行手順

Jupyter からリポジトリのノートブックを開いてください。各ノートブックは `pyprojroot` でルートを特定するため、ノートブックのカレントディレクトリを手動で切り替える必要はありません。処理は次の順で進めます。

### 1. 合成雑音を作成

`notebooks/1.Synthesize_data/make_synthesized_data.ipynb` を実行します。現在のノートブックの設定では、サンプリング周波数 16 kHz、長さ 10 秒、帯域 20–7,980 Hz の信号を次の場所に生成します。

- `Dataset/train/`：2,000 件
- `Dataset/validate/`：2,000 件
- `Dataset/test/`：5 件（テスト用セルは 5 回、1 件ずつ生成）

テスト用セルは合成信号を作る設定です。NOISEX-92 や実録騒音のダウンロード・配置は行いません。

### 2. RIR を作成

`notebooks/1.Synthesize_data/make_rir.ipynb` を実行します。室寸法や RT60 の設定に基づいて `.mat` ファイルを生成します。既定では `rirs/rt60_0.15s/` から `rirs/rt60_0.25s/` までの各フォルダーに、一次経路 `Primary_path.mat` と二次経路 `Secondary_path.mat` を保存します。

### 3. モデルを学習

`notebooks/2.Train_Network/Train_Network.ipynb` を実行します。`Dataset/train/`、`Dataset/validate/` と `rirs/` を使い、既定では 30 エポック学習します。検証 NMSE が更新された時点の重みを次のファイルに保存します。

```text
models/synth_delay_1.pth
```

学習コードは利用可能な場合に CUDA を使い、GPU が複数ある場合は DataParallel を有効にします。

### 4. ANC 推論・評価

`notebooks/3.Noise_Cancellation_Code/Noise_Cancellation.ipynb` を実行します。設定してある RIR フォルダー、テスト WAV、モデルのパスやファイル名を、実際に用意したものに合わせてください。既定の入力 WAV 名は `1_from_1733_to_2243_Hz.wav` です。合成ノートブックの生成ファイル名と一致しないことがあるため、`Dataset/test/` にある実ファイル名へ `sound_name` を変更してください。

このノートブックは `models/synth_delay_1.pth` と指定した RIR を読み込み、ANC オフ・オンの波形比較、NMSE の計算、`GFANC.png` の出力を行います。実録騒音を評価する場合は、利用許諾のある WAV を用意し、そのファイル名に設定してください。`loading_real_wave_noise` は音源を読み込み 16 kHz にリサンプリングします。

## NOISEX-92

テストデータとして NOISEX-92 を利用する場合、データは別途入手し、必要な音源をノートブックが読む WAV 形式・配置に合わせて準備してください。このリポジトリにはデータ本体や自動取得処理は含まれていません。

データセットの参照先：[Speech and noise Corpora for Pitch Estimation of Human Speech (Zenodo)](https://zenodo.org/records/3921794)

## パッケージの利用

インストール後は、`deepanc` 以下のモジュールをインポートできます。

```python
from deepanc.synthesize.Synthesize_data import SoundGenerator
from deepanc.train.Train import train_validate
from deepanc.inference.loading_real_wave_noise import loading_real_wave_noise
```

## 再現性に関する注意

合成ノートブックによる学習は、用意した WAV にシミュレーション RIR を適用する実験フローです。実録騒音を使う論文条件の再現には、論文で指定された音源、データ分割、収録・評価条件を別途確認して準備してください。既定の合成データ生成数や RIR 条件はノートブック内で変更できます。

