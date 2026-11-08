# DeepANC

DeepANC 論文の手法を PyTorch で実装・実験する研究プロジェクトです。合成雑音および Freesound 音源の準備、室内インパルス応答（RIR）の生成、モデル学習、能動騒音制御（ANC）の評価を行います。

本リポジトリには、論文で提案された手法の実験コードと、本リポジトリで整備した学習・データ準備処理を含みます。論文由来の手法と、本リポジトリで実装・追加した部分を区別してご覧ください。

## 提供モデル

GitHub リポジトリの `models/` に学習済み重みを含めています。

| ファイル | 学習データ | 設定 |
| --- | --- | --- |
| `models/freesound_delay_1.pth` | Freesound 音源 | delay = 1 |
| `models/synthesized_delay_1.pth` | 合成バンド制限雑音 | delay = 1 |

モデル構造は `src/deepanc/common/Network.py` の `CRN` クラスで定義し、`ModelType`で接続方式を選択します。

| `ModelType` | 構成 |
| --- | --- |
| `ModelType.NONE` | エンコーダーとデコーダー間の接続なし |
| `ModelType.SKIP` | スキップ接続 |
| `ModelType.RES` | エンコーダー特徴を加算する残差接続 |

```python
from deepanc.common.Network import CRN, default_channels, ModelType

model = CRN(default_channels, ModelType.SKIP)
```

チェックポイントを別のモデル構成や前処理と組み合わせる場合は、学習時の設定と一致することを確認してください。

## プロジェクト構成

```text
.
├── models/                           # 学習済み重み
├── Datasets/
│   └── statistics/                   # Freesound 音源のマニフェスト CSV
├── Dataset/                          # 合成音声データ
├── rirs/                             # 生成した室内インパルス応答
├── history/                          # 学習損失・NMSE の履歴 CSV
├── NOISEX92/                         # ANC 評価用音声（別途取得）
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

Notebook は `pyprojroot.here()` でプロジェクトルートを探します。インストールした環境を Jupyter のカーネルとして選択してください。

## 主な機能

- DeepANC の CRN、STFT/ISTFT、音響経路の畳み込み、NMSE 評価
- `ModelType`によるNONE・SKIP・RES構成の切り替え
- Freesound 音源の取得と FLAC 形式への変換
- Freesound 音源一覧の CSV マニフェスト化
- 可変長音声のバッチ化
- 学習・検証損失と NMSE の CSV 保存
- 学習済み重みの保存と配布

手法の出典は [Deep ANC: A deep learning approach to active noise control](https://www.sciencedirect.com/science/article/pii/S0893608021001258) です。Freesound 音源の取得には [construct-dataset-from-freesound](https://github.com/koba1in/construct-dataset-from-freesound) を使用します。

## 必要環境とインストール

- Python 3.11 以上
- `uv` または `pip`
- Freesound 音源の取得には Hugging Face Read トークン
- GPU 学習を行う場合は、環境に合った CUDA 対応 PyTorch

`pyproject.toml`の`normal`と`titanx`は異なる PyTorch / torchaudio バージョンを指定します。使用環境に合うextraを選んでください。両方を同時に使わないでください。

プロジェクトルートで通常の依存関係と分析 Notebook 用依存関係をインストールします。

```bash
uv pip install -e ".[normal,polars]"
uv pip install seaborn
```

`pip`を使う場合:

```bash
python -m pip install -e ".[normal,polars]"
python -m pip install seaborn
```

Freesound 分析 Notebook は`polars`と`seaborn`も使用します。

## データセット

### Freesound 音源（学習用）

取得ツールは Hugging Face の [benjamin-paine/freesound-laion-640k](https://huggingface.co/datasets/benjamin-paine/freesound-laion-640k) をストリーミングで読み込みます。取得前に Hugging Face Read トークンを用意し、環境変数 `HF_TOKEN` に設定してください。

macOS / Linux:

```bash
export HF_TOKEN="hf_..."
```

Windows PowerShell:

```powershell
$env:HF_TOKEN = "hf_..."
```

プロジェクトルートで、学習 split の CC0 音源を取得します。

```bash
construct-dataset-from-freesound --split train --path ./Datasets --license CC0-1.0
```

音声は`Datasets/train/`に`<freesound_id>.flac`として保存されます。出力は16 kHz、モノラル、16-bit PCMのFLACです。同名ファイルはスキップされます。不完全なファイルが残った場合は削除してから再実行してください。

Freesound 音源には作品ごとのライセンスが適用されます。再配布や学習済みモデルの公開を行う前に、利用した音源の条件やCC-BYの帰属表示義務を確認してください。

### Speech and Noise Corpora（評価用データセット参照先）

音声評価用データセットの参照先として、Zenodo の [Speech and Noise Corpora for Pitch Estimation of Human Speech](https://zenodo.org/records/3921794) を掲載しています。このレコードには複数の音声・雑音コーパスが含まれ、その中に`NOISEX92.zip`があります。本プロジェクトの評価NotebookはNOISEX-92音源を参照します。Zenodoから必要なデータを取得し、展開後、評価Notebookが参照するフォルダー構成に合わせて配置してください。

このZenodoレコードはピッチ推定評価向けのコーパス集です。本プロジェクトでは収録されているNOISEX-92の音源をANC評価に利用する想定です。各コーパスの利用条件は異なるため、ダウンロード・利用・再配布の前に個別の条件を確認してください。

### Freesound マニフェスト

`notebooks/1.Synthesize_Download_data/analyze_freesound.ipynb`は`Datasets/train/`の音源を調べ、`Datasets/statistics/`に`train.csv`と`validation.csv`を作成します。現在のNotebook設定では、録音時間を基準に次の範囲で選びます。

- 学習用: 6～12秒
- 検証用: 20.0～20.1秒

この設定では学習用と検証用で録音時間の分布が異なります。データや実験目的に応じて範囲を見直してください。`verify_train.csv`と`verify_validation.csv`は動作確認用の少数音源を指すマニフェストです。

## プログラムの実行手順

以下は、Freesound 音源でデータ準備から学習・評価まで進める基本手順です。コマンドとNotebookはプロジェクトルートから実行してください。

1. **依存関係をインストールする**  
   「必要環境とインストール」の手順を実行します。Freesound音源を取得する場合は`HF_TOKEN`も設定します。
2. **Freesound 音源を取得する**  
   「データセット」節のコマンドを実行し、`Datasets/train/`に音源を用意します。
3. **マニフェストを作成する**  
   `notebooks/1.Synthesize_Download_data/analyze_freesound.ipynb`を上から実行し、`Datasets/statistics/train.csv`と`validation.csv`を作成します。
4. **RIR を生成する**  
   `notebooks/1.Synthesize_Download_data/make_rir.ipynb`を実行します。生成したRIRは`rirs/`に配置されます。
5. **モデルを学習する**  
   `notebooks/2.Train_Network/Train_Network.ipynb`または`notebooks/2.Train_Network/Train_Network.py`を使います。標準の学習スクリプトは`CRN(default_channels, ModelType.SKIP)`を使用します。別の接続方式を学習する場合は、モデル生成箇所の`ModelType.SKIP`を`ModelType.NONE`または`ModelType.RES`に変更してください。

   Notebookの`verify`は現在`True`なので、動作確認用マニフェストで学習します。全マニフェストで学習する場合は`False`に変更してください。学習設定は`src/deepanc/train/config.py`の`TrainConfig`で管理します。
6. **ANC 評価を実行する**  
   NOISEX-92の音源を`NOISEX92/<noise_name>/signal.wav`の形式で配置し、`notebooks/3.Noise_Cancellation_Code/Noise_Cancellation.ipynb`を実行します。Notebookは評価音源ごとにNMSEを出力します。Notebook内の`ModelType`と重みファイルが同じ構成に対応していることを確認してください。

Notebookを起動する例:

```bash
jupyter lab
```

合成雑音で学習する場合は、先に`notebooks/1.Synthesize_Download_data/make_synthesized_data.ipynb`を実行し、`Dataset/`以下に合成音声を生成してください。現在の設定ではtrain 2,000件、validate 1,000件、test 5件の、16 kHz・10秒のバンド制限雑音を作成します。Freesound用の`Datasets/`と合成音声用の`Dataset/`は別のディレクトリです。

## 学習出力

Freesound学習では、履歴を`history/freesound_delay_1.csv`、重みを`models/freesound_delay_1.pth`に保存する設定です。学習スクリプトの設定により、モデル種別・遅延・バッチサイズ・デバイス・エポック数・履歴ファイルのパスなどを指定します。

Notebookの確認実行と本学習は同じ出力先を使う設定です。確認用学習が既存の重みや履歴を上書きしないよう、実行前に保存先を確認してください。実験ごとに出力先を分ける場合は、重みと履歴を同じ実験ディレクトリに保存してください。

## ANC 評価

`notebooks/3.Noise_Cancellation_Code/Noise_Cancellation.ipynb`は`NOISEX92/<noise_name>/signal.wav`形式の入力を読み込み、指定したRIRと学習済みモデルでANCを評価します。Notebookでは`ModelType.SKIP`、`ModelType.NONE`、`ModelType.RES`の構成を選択できます。使用する重みも選択したモデル構成に対応させてください。Freesound取得ツールのHugging Face `test` splitとNOISEX-92は別のデータです。

## データ、モデル、ライセンス

音声データや生成したRIRはローカルで用意します。Freesound音源には作品ごとのライセンスが適用されます。再配布や学習済みモデルの公開を行う前に、利用した音源のライセンス条件やCC-BYの帰属表示義務を確認してください。NOISEX-92を含むZenodo収録データも、コーパスごとの利用条件を確認してください。

`.gitignore`は一般の生成チェックポイント`*.pth`を除外します。GitHubに登録済みの`models/freesound_delay_1.pth`と`models/synthesized_delay_1.pth`は、このリポジトリで配布する重みです。重みを更新する場合は、Git上で追跡対象になっていることを確認してください。音源・RIR・モデルの再配布条件も個別に確認してください。

## 既知の制約

- Freesound音源は可変長です。現行のcollate処理はバッチ内でゼロ埋めしますが、損失・NMSEでは元の音声長に応じたマスクを適用しません。音源長の異なるデータを比較する際は、この影響を考慮してください。
- `istft`はSTFTフレーム長に合わない入力で末尾をゼロ埋めする場合があります。評価Notebookに警告が出る場合は、出力長と対象区間を確認してください。
- Freesound音源、NOISEX-92、RIR生成結果はリポジトリに含まれていません。再現には各データの準備が必要です。
