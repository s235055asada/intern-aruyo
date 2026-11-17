# Hermes Agent型

Markdown化した会議ログをローカルフォルダから検索し、GeminiまたはOpenAIで
根拠付き回答を生成するCLIチャットです。Windows PowerShellでの利用を想定しています。

Slack Socket Modeにも対応しています。公開Webサーバーを用意せず、チャンネルで
Botをメンションして会議ログへ質問できます。設定は
[`SLACK_SETUP.md`](./SLACK_SETUP.md)を参照してください。

Slack版は資料を起動時にメモリへ読み込み、24時間経過後の質問をきっかけに
バックグラウンドで更新します。既定モデルは`gemini-2.5-flash-lite`、回答待機上限は16秒です。
曖昧な改善相談には、職種と相談テーマの根拠に基づく「提案」と確認質問を返します。

## 参照するMDフォルダ

参照先は [`config.json`](./config.json) の `local_docs_dir` で指定します。
現在の設定は次のとおりです。

```json
{
  "local_docs_dir": "..\\MD"
}
```

相対パスは`config.json`がある`Hermes Agent型`フォルダを基準に解決されます。
現在の配置では、実際に参照するフォルダは次です。

```text
任意の親フォルダ\MD
```

推奨フォルダ構成:

初期設定ではリポジトリ直下の`MD`を参照します。Google Driveを使う場合は、
`local_docs_dir`を同期フォルダへ変更し、パソコン版Googleドライブを起動してください。
以下はローカル保存時の構成例です。

```text
任意の親フォルダ\
├─ Hermes Agent型\
│  ├─ hermes_agent.py
│  ├─ config.json
│  └─ system.md
└─ MD\
   ├─ meeting-1.md
   └─ meeting-2.md
```

- `MD`配下はサブフォルダも再帰的に検索します。
- 対応形式は`.md`、`.markdown`、`.txt`です。
- 同じフォルダに同名の`.md`と`.txt`がある場合は`.md`を優先します。
- Google DriveのURLを直接読むのではなく、PCへ同期・保存されたローカルファイルを読みます。

別の場所を参照する場合は、`config.json`を変更します。

```json
{
  "local_docs_dir": "D:\\CompanyData\\MeetingLogs"
}
```

一度だけ別フォルダを使う場合は、設定ファイルを変えずに指定できます。

```powershell
python .\hermes_agent.py --chat --provider gemini --search keyword `
  --docs-dir "D:\CompanyData\MeetingLogs"
```

## APIキーを設定する

Geminiを使う場合は`GEMINI_API_KEY`を設定します。APIキーを`README.md`、
`config.json`、ソースコード、質問データへ直接書かないでください。

### 一時的に設定する

現在開いているPowerShellだけで有効になります。PowerShellを閉じると消えます。

```powershell
$env:GEMINI_API_KEY="ここにGemini APIキー"
```

### Windowsユーザー環境変数へ保存する（推奨）

次のコマンドを1回実行します。

```powershell
[Environment]::SetEnvironmentVariable(
    "GEMINI_API_KEY",
    "ここにGemini APIキー",
    "User"
)
```

設定後はPowerShellを閉じ、新しいPowerShellを開いてください。値を表示せずに
設定済みか確認できます。

```powershell
[bool]$env:GEMINI_API_KEY
```

`True`なら設定済みです。OpenAIを使う場合は同じ方法で`OPENAI_API_KEY`を
設定します。APIの無料枠・料金・利用上限は各APIプロジェクトの設定に依存します。

現在のテスト環境では、環境変数にキーがない場合だけ`目標１.txt`末尾のGeminiキーを
参照します。この方法はテスト期間限定です。他人へ共有する前にキーを文書から削除し、
環境変数だけを使う構成へ戻してください。

## 別のPCへ導入する

前提はPython 3.11以降とPowerShellです。GBrainは任意機能なので、最初は
`keyword`検索で動作確認します。

1. `Hermes Agent型`フォルダを新しいPCへ配置する。
2. 会議ログを`Hermes Agent型`と同じ親フォルダの`MD`へ配置する。または
   `config.json`の`local_docs_dir`を実際の保存先へ変更する。
3. PowerShellで次を実行する。

```powershell
cd "D:\任意の場所\Hermes Agent型"

py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r .\requirements.txt
```

PowerShellで仮想環境を有効化できない場合は、現在のPowerShellだけ実行を許可します。

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```

4. 前節の方法で`GEMINI_API_KEY`をWindowsユーザー環境変数へ設定し、
   新しいPowerShellを開く。
5. APIを呼ばず、MD検索だけを確認する。

```powershell
cd "D:\任意の場所\Hermes Agent型"
.\.venv\Scripts\Activate.ps1
python .\hermes_agent.py "会議で決まったことは？" `
  --provider gemini --search keyword --show-search-results
```

検索結果にMDファイル名が表示されたら、チャットを起動します。

```powershell
python .\hermes_agent.py --chat --provider gemini --search keyword
```

`keyword`はGBrain、Bun、Embedding索引がなくても動作します。GBrainを導入済みの
PCでは`--search enhanced`を使用できますが、速度と再現性を優先する通常利用では
`keyword`を推奨します。

## このPCで最短起動する

新しいPowerShellを開き、次を実行します。APIキーをWindowsユーザー環境変数へ
登録済みなら、起動時にキーを入力する必要はありません。

```powershell
cd "C:\aruyoインターン\Hermes Agent型"
python .\hermes_agent.py --chat --provider gemini --search keyword
```

次の表示が出たら起動完了です。

```text
会議ログAIチャットを開始しました。
終了: /exit  履歴を消去: /clear  ヘルプ: /help

あなた>
```

`あなた>` の後ろへ質問を入力してEnterを押します。終了するときは`/exit`を
入力します。

起動後の質問方法、回答の見方、終了方法、エラー対応は
[`USAGE.md`](./USAGE.md) を参照してください。

## ファイル

- `system.md`: Hermes Agent に渡すシステムプロンプト
- `hermes_agent.py`: ローカルの Markdown / txt を検索して回答するCLI
- `slack_bot.py`: Slack Socket Modeで検索・回答処理を呼び出すBot
- `slack_manifest.yaml`: Slack App作成時に読み込む権限・イベント設定
- `SLACK_SETUP.md`: Slack管理画面、トークン、チャンネル、起動の手順
- `config.json`: MD保存元のDrive URLと、実際に読むローカルフォルダの設定
- `USAGE.md`: 起動してから終了するまでの日常操作ガイド

## 使い方

1. Google Drive の対象フォルダをPCに同期する、または任意のローカルフォルダへMarkdownを保存する。
   - Drive URL: `https://drive.google.com/drive/folders/YOUR_FOLDER_ID`
   - この実行環境からはDriveの認証情報がないため、非公開フォルダを直接読むことはできません。
   - `config.json` の `local_docs_dir` に、同期/保存したMarkdownフォルダのパスを書いてください。
2. Geminiを使う場合は `GEMINI_API_KEY`、OpenAIを使う場合は `OPENAI_API_KEY` を環境変数に設定する。
3. 以下のように実行する。

```powershell
python .\hermes_agent.py "AruyoのSlackボット開発で決まったことは？" --docs-dir "C:\path\to\markdown"
```

`config.json` の `local_docs_dir` を設定済みなら、`--docs-dir` は省略できます。

```powershell
python .\hermes_agent.py "AruyoのSlackボット開発で決まったことは？"
```

モデルを変える場合:

```powershell
python .\hermes_agent.py "質問文" --docs-dir "C:\path\to\markdown" --model gpt-4.1-mini
```

Gemini APIを使う場合:

```powershell
$env:GEMINI_API_KEY="あなたのGemini APIキー"
python .\hermes_agent.py "質問文" --provider gemini
```

既定のGeminiモデルは `gemini-2.5-flash` です。

## 精度改善版の検索

既定では、速度と再現性を優先してローカルの日本語BM25 `keyword` 検索を使います。
現在の疑似未知16問ではFile/Evidence Top-1が87.5%、Top-5が93.8%です。

```powershell
python .\hermes_agent.py "オンボーディングについて決まったことを教えて" --provider gemini --top-k 8
```

検索APIを使わないローカルBM25だけで実行する場合:

```powershell
python .\hermes_agent.py "質問文" --provider gemini --search keyword
```

検索結果だけを確認する場合:

```powershell
python .\hermes_agent.py "オンボーディングについて決まったことを教えて" --provider gemini --show-search-results
```

Gemini Embeddingsによる文書単位の意味検索も併用する場合:

```powershell
python .\hermes_agent.py "質問文" --provider gemini --search hybrid
```

課金なしのローカル多言語Embeddingを使う場合:

```powershell
python .\hermes_agent.py "質問文" --provider gemini --search local
```

`local` の初回だけ約220MBのモデルをダウンロードします。作成した
989チャンクの索引は `.cache/local_chunk_embeddings.json` に保存されます。
ローカルEmbedding単体より、日本語BM25の方が現在の評価では高精度です。

## GBrainを使った言い換え検索

GBrain `0.42.73.2` とローカルPGlite DBを導入し、46ファイル・1,283チャンクの
Gemini Embedding索引を `.gbrain` に作成済みです。

GBrain単体で検索する場合:

```powershell
python .\hermes_agent.py "質問文" --provider gemini --search gbrain
```

BM25のTop-5を維持しながら、GBrainが高い確信を持つ場合だけ順位を補正する場合:

```powershell
python .\hermes_agent.py "質問文" --provider gemini --search enhanced
```

`enhanced` は未知言い換え10問でFile Top-1を80%から90%、Evidence Top-1を
60%から90%へ改善し、Top-5はどちらも100%を維持しました。既存20問では
Evidence Top-1を75%から85%へ改善しました。

GBrain検索にはBun、GBrain本体、`.gbrain` DB、Gemini APIキーが必要です。
検索ごとに質問のGemini Embeddingを作成するため、APIキーのプロジェクトが
Paid Planの場合は料金が発生する可能性があります。検索API呼び出しを避ける場合は
`--search keyword` を指定してください。成功した検索結果は
`.cache/gbrain_queries.json`へ保存され、同じ文書・質問・件数の再検索ではAPIを
呼ばずキャッシュを使います。

## 嘘を抑える根拠検証

回答内の事実には `[S1]` のような根拠IDを必須にしています。AIが根拠IDを
付けなかった断定や、存在しないIDを付けた行は、回答表示前にプログラムが削除します。
引用先に存在しない数値・日付・人名を含む行も削除します。根拠内の数値から加算・減算・
乗算で算出できる値は許可します。
根拠欄はAIの自由記述を使わず、実際に検索されたMDファイル名、チャンク番号、原文抜粋を
自動表示します。検索結果が質問への答えを直接含まない場合は
「会議ログ内では確認できません」と回答します。

会議全体の要約では、対象会議を時系列に最大6区間へ分け、区間ごとに`S1`から
`S6`の根拠IDを付けます。全文を1つの根拠として扱わないため、決定事項や次の作業が
会議のどの範囲にあるかを確認できます。

金額、日時・期限、人数・件数、担当者を尋ねる質問は、質問トピックと必要な値が
検索結果に存在するかを回答生成前に確認します。根拠が不足する場合はGemini APIを
呼ばずに回答を拒否します。

この処理は作話の可能性を大幅に下げますが、引用した原文の解釈そのものが必ず正しいと
数学的に保証するものではありません。重要な判断では表示された原文も確認してください。

## チャットとして会話する

次のコマンドで、終了するまで何度でも質問できるチャットモードを開始します。

```powershell
cd "C:\aruyoインターン\Hermes Agent型"
python .\hermes_agent.py --chat --provider gemini
```

実行例:

```text
あなた> オンボーディングについて決まったことを教えて
AI> ...

あなた> その担当者は誰？
AI> ...

あなた> 期限は？
AI> ...
```

チャット中は直近の会話履歴と、直前回答が引用した会議ファイル名を検索へ引き継ぐため、
「その会議」「その担当者」「期限は？」のような続きの質問ができます。利用できる
コマンドは次のとおりです。

- `/clear`: 現在の会話履歴を消去する
- `/help`: 操作コマンドを表示する
- `/exit`: チャットを終了する

最初の質問を付けて起動することもできます。

```powershell
python .\hermes_agent.py "オンボーディングについて教えて" --chat --provider gemini
```

通常は既定の `keyword` を使うため `--search` は省略できます。同じファイルの
隣接チャンクが検索結果を占有しないよう、ファイル単位で検索結果を分散します。
採用ログで頻出する「選考→先行・専攻」「面接官→先行感・先行缶」などの例は、原文を
変更せず検索時の同義語として補います。
回答には会議名、日付、発言者のメタデータを渡し、次の項目を分けて出力します。

- 結論
- 決定事項
- 次の作業
- 担当者
- 期限
- 未決事項
- 根拠

## 検索精度の評価

`evaluation_questions.json` の固定質問セットを使い、期待する会議ファイルが
検索上位5件に含まれる割合を比較します。

```powershell
python .\evaluate_retrieval.py --mode keyword
python .\evaluate_retrieval.py --mode gbrain
python .\evaluate_retrieval.py --mode enhanced
python .\evaluate_retrieval.py --mode hybrid
python .\evaluate_retrieval.py --mode both
```

質問と正解ファイルは、実際の利用例が増えたら
`evaluation_questions.json` に追加してください。GBrainを評価する際も同じ質問を使います。

内容ベースの20問で、正解ファイルだけでなく回答に必要な根拠チャンクまで
評価する場合:

```powershell
python .\evaluate_retrieval.py `
  --mode both `
  --questions .\accuracy_questions.json `
  --top-k 5
```

調査結果は `accuracy_report.md` に記録しています。

回答可能性の事前判定を、回答可能5問・回答不可5問で評価する場合:

```powershell
python .\evaluate_answerability.py
```

この評価は`keyword`検索だけを使い、Gemini APIを呼びません。現在の10問では
10/10（100%）です。質問は`answerability_questions.json`へ追加できます。

既存評価で使っていない会議ログから疑似未知質問を生成して評価する場合:

```powershell
python .\generate_synthetic_questions.py --count 20
python .\evaluate_retrieval.py `
  --mode both `
  --questions .\synthetic_questions.json `
  --top-k 5
```

疑似16問の結果と注意点は `synthetic_evaluation_report.md` に記録しています。

会議全体の要約質問を評価する場合:

```powershell
python .\evaluate_retrieval.py `
  --mode both `
  --questions .\summary_questions.json `
  --top-k 6
```

要約質問では、上位3件の候補会議を全文コンテキストに切り替えます。
「2回目」「-3」のような回数指定は該当回を優先し、回数指定なしで同名候補が
複数ある場合は、内容を混ぜずに対象ファイルの確認を返します。
7問での改善結果と同名会議の注意点は `summary_evaluation_report.md` に記録しています。
この検索評価はGemini APIを呼ばないため、API料金は発生しません。

Gemini APIを使った実務想定の会話検証結果は
`api_conversation_evaluation_report.md` に記録しています。
MD更新後の再検証結果は `revalidation_2026-08-14.md` に記録しています。

未知の言い換え10問を評価する場合:

```powershell
python .\evaluate_retrieval.py `
  --mode both `
  --questions .\accuracy_holdout_questions.json `
  --top-k 5
```

## 目標1の流れ

文字起こしをMarkdown化し、ローカルまたはGoogle Drive同期フォルダに保存する。その保存先をHermes Agentが読み、会議内容に関する質問へ回答する。最終的にはSlackチャットボット化するが、まずはSlack連携なしでこの質問応答システムを動かす。
