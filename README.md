# DeepANC

DeepANC 論文の手法を PyTorch で実装・実験するための研究プロジェクトです。合成雑音および Freesound 音源の準備、室内インパルス応答（RIR）の生成、モデル学習、能動騒音制御（ANC）の評価を行います。

本リポジトリは、論文で提案された手法の実験コードと、このリポジトリで整備した学習・データ準備の処理を含みます。手法自体の提案者と、本リポジトリでの実装・追加部分を区別してご覧ください。

## 提供モデル

GitHub リポジトリの `models/` には、学習済み重みを含めています。

| ファイル | 学習データ | 設定 |
| --- | --- | --- |
| `models/freesound_delay_1.pth` | Freesound 音源 | delay = 1 |
| `models/synthesized_delay_1.pth` | 合成バンド制限雑音 | delay = 1 |

モデルは `src/deepanc/common/Network.py` の CRN 構成と組み合わせて使用します。チェックポイントを別のモデル構成や前処理と組み合わせる場合は、学習時の設定と一致することを確認してください。

## プロジェクト構成

```text
.
├── models/                           # GitHub に登録された学習済み重み
├── Datasets/
│   └── statistics/                   # Freesound 音源のマニフェスト CSV
├── Dataset/                          # 合成音声データ
├── rirs/                             # 生成した室内インパルス応答
├── history/                          # 学習損失・NMSE の履歴 CSV
├── NOISEX92/                         # 最終評価用音声（別途用意）
├── notebooks/
│   ├── 1.Synthesize_Download_data/
│   ├── 2.Train_Network/
│   └── 3.Noise_Cancellation_Code/
├── src/deepanc/
│   ├── common/                       # データローダー、モデル、信号処理
│   ├── synthesize/                   # 合成雑音の生成
│   ├── train/                        # 学習・検証処理
│   └── inference/                    # ANC 推論、FxLMS、音声読み込み
├── tests/
└── pyproject.toml
```

Notebook は `pyprojroot.here()` でプロジェクトルートを探します。リポジトリをインストールした環境の Jupyter カーネルを選択して実行してください。

## 主な変更・機能

- 論文の DeepANC 手法を CRN、STFT/ISTFT、音響経路の畳み込み、NMSE 評価に分けて実装
- Freesound 音源を取得する CLI を別リポジトリとして用意し、DeepANC から利用
- Freesound のファイル一覧を CSV マニフェストで管理し、FLAC 音源を学習データとして読み込み
- 可変長音声をバッチ化する collate 処理を追加
- 学習・検証の損失と NMSE を `history/` に保存
- 学習済み重みを `models/` に提供

手法の出典は [Deep ANC: A deep learning approach to active noise control](https://www.sciencedirect.com/science/article/pii/S0893608021001258) です。Freesound 音源の取得には [construct-dataset-from-freesound](https://github.com/koba1in/construct-dataset-from-freesound) を使用します。

## 必要環境とインストール

- Python 3.10 以上
- uv または pip
- Freesound データセットの取得には Hugging Face Read トークン
- GPU 学習を行う場合は、環境に適合する CUDA 対応 PyTorch

通常の依存関係、Freesound 取得 CLI、および分析 Notebook 用の Polars をインストールする例です。

```bash
uv pip install -e ".[normal,polars]"
uv pip install seaborn
```

pip を使う場合:

```bash
python -m pip install -e ".[normal,polars]"
python -m pip install seaborn
```

`normal` と `titanx` は異なる PyTorch / torchaudio バージョンを指定するため、同時に選ばず、利用環境に合う方を使ってください。Freesound 分析 Notebook は `polars` と `seaborn` も使用します。

## Freesound 音源の取得と準備

取得ツールは Hugging Face の [benjamin-paine/freesound-laion-640k](https://huggingface.co/datasets/benjamin-paine/freesound-laion-640k) をストリーミングで読み込みます。トークンを環境変数 `HF_TOKEN` に設定してください。

macOS / Linux:

```bash
export HF_TOKEN="hf_..."
```

Windows PowerShell:

```powershell
$env:HF_TOKEN = "hf_..."
```

プロジェクトルートから、例として CC0 と CC BY 4.0 の音源を取得します。

```bash
construct-dataset-from-freesound \
  --split train test \
  --path ./Datasets \
  --license CC0-1.0 CC-BY-4.0
```

出力は次の形式です。

```text
Datasets/
├── train/
│   └── <freesound_id>.flac
└── test/
    └── <freesound_id>.flac
```

音声は 16 kHz、モノラル、16-bit PCM の FLAC として保存されます。取得ツールは同名ファイルをスキップします。不完全なファイルが残った場合は、そのファイルを削除してから再取得してください。

### マニフェストの作成

`notebooks/1.Synthesize_Download_data/analyze_freesound.ipynb` は `Datasets/train/` の音源を調べ、`Datasets/statistics/` に `train.csv` と `validation.csv` を作成します。現在のNotebook設定では、録音時間によって次のように選びます。

- 学習用: 6～12 秒
- 検証用: 20.0～20.1 秒

この設定では、学習用と検証用の録音時間の分布が異なります。データや実験目的に合わせて範囲を見直してください。確認実行用の `verify_train.csv` と `verify_validation.csv` は、それぞれ少数の音源を指すマニフェストです。

## 合成雑音と RIR の生成

### 合成雑音

`notebooks/1.Synthesize_Download_data/make_synthesized_data.ipynb` は次の合成音声を作ります。

- `Dataset/train/`: 2,000 件
- `Dataset/validate/`: 1,000 件
- `Dataset/test/`: 5 件
- 16 kHz、10 秒、20～7,980 Hz のバンド制限雑音

Freesound 学習のNotebookは `Datasets/train/` と `Datasets/statistics/*.csv` を使用します。合成音声は別の `Dataset/` 以下に保存されるため、保存先を混同しないでください。

### RIR

`notebooks/1.Synthesize_Download_data/make_rir.ipynb` は、既定の室寸法と RT60 条件から、`rirs/rt60_0.15s/` から `rirs/rt60_0.25s/` までの経路を生成します。

**実行前の修正:** 現在のNotebookには `save_dir.mkdir(parents=True, exists_ok=True)` という引数名の誤記があります。実行セルでは `exists_ok=True` を `exist_ok=True` に直してください。

## 学習

主な学習入口は次の2つです。

- `notebooks/2.Train_Network/Train_Network.ipynb`
- `notebooks/2.Train_Network/Train_Network.py`

現在の Freesound 学習設定は `Datasets/train/` を音源フォルダーとして、`Datasets/statistics/train.csv` と `validation.csv` のマニフェストで学習・検証対象を選びます。RIR は `rirs/` から読み込み、学習履歴を `history/freesound_delay_1.csv`、重みを `models/freesound_delay_1.pth` に保存します。

Notebook の `verify` は現在 `True` です。この状態では確認用マニフェストを使って少数音源だけを学習します。全マニフェストで学習する場合は `verify = False` にしてください。確認実行と本学習で保存先が同じため、確認モデルで既存の重みを上書きしないよう、必要なら保存先を変更してください。

学習時の設定は `src/deepanc/train/config.py` の `TrainConfig` で管理します。モデル名、遅延、バッチサイズ、デバイス、エポック数、履歴ファイルのパスを設定できます。

## ANC 評価

`notebooks/3.Noise_Cancellation_Code/Noise_Cancellation.ipynb` は、現在 `NOISEX92/<noise_name>/signal.wav` 形式の入力を読み込み、指定した RIR と `models/freesound_delay_1.pth` を用いて ANC を評価します。Notebook は評価音源ごとに NMSE を出力します。

NOISEX-92 の音源はリポジトリに含めていません。利用条件を確認のうえ別途入手し、Notebookが参照するフォルダー構成に配置してください。Freesound 取得ツールの `test` split と NOISEX-92 は別のデータです。現行の評価 Notebook が参照するのは NOISEX-92 です。

## データ、モデル、ライセンス

音声データや生成した RIR はローカルで用意します。Freesound 音源には作品ごとのライセンスが適用されます。再配布や学習済みモデルの公開を行う前に、対象音源のライセンス条件と CC-BY の帰属表示義務などを確認してください。

`.gitignore` は一般の生成チェックポイント `*.pth` を除外します。GitHub に登録済みの `models/freesound_delay_1.pth` と `models/synthesized_delay_1.pth` は、このリポジトリで配布する学習済み重みです。重みを更新する場合は、Git上で追跡対象になっていることを確認してください。音源・RIR・モデルの再配布条件も個別に確認してください。

## 既知の制約

- Freesound 音源は可変長です。現行の collate 処理はバッチ内でゼロ埋めしますが、損失・NMSEは元の音声長でマスクされません。音源長の異なるデータを比較する際は、この影響を考慮してください。
- `istft` はSTFTフレーム長に合わない入力で末尾をゼロ埋めする場合があります。評価Notebookに警告が出る場合、出力長と対象区間を確認してください。
- Freesound の配布物、NOISEX-92、RIR生成結果はこのリポジトリに含まれていません。再現には各データの準備が必要です。
 
