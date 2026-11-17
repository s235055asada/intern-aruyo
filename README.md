# intern-aruyo

会議ログの前処理、検索、根拠付き回答、Slack連携を行うコードです。
「Hermes Agent型」は独自実装の名称で、外部のHermes Agent本体は使用していません。

## 構成

- `Hermes Agent型/`: 会議ログの検索・回答、Slack Socket Mode Bot、評価スクリプト、単体テスト
- `RAG型/`: 会議テキストの整形・分割・タグ付けのコード

APIキー、Slackトークン、会議原文、実データを含む評価質問・評価レポート、
検索キャッシュ、実行ログは含めていません。
評価スクリプトを使う場合は、権限のある評価データを別途用意してください。

## Slack Botの起動

Python 3.11以降を使用します。PowerShellでリポジトリのフォルダを開いて実行します。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install slack-bolt
cd "Hermes Agent型"

$env:GEMINI_API_KEY="取得したGemini APIキー"
$env:SLACK_BOT_TOKEN="取得したBot User OAuth Token"
$env:SLACK_APP_TOKEN="取得したApp-Level Token"
$env:SLACK_CHANNEL_ID="対象チャンネルID"

python .\slack_bot.py
```

会議ログの`.txt`または`.md`は、リポジトリ直下に作成した`MD`フォルダへ配置します。
Google Drive同期フォルダを読む場合は、`Hermes Agent型/config.json`の
`local_docs_dir`を、そのPCのフォルダパスへ変更してください。

Slack Appの作成・権限・招待手順は
[SLACK_SETUP.md](./Hermes%20Agent型/SLACK_SETUP.md)を参照してください。
稼働させるPCまたはサーバーが必要です。n8n Cloudへの移行は未実装です。

## 現在の動作

- ローカルBM25検索を既定として使用。GBrainは任意の検索方式
- 資料を起動時にメモリへ読み込み、24時間経過後の質問をきっかけにバックグラウンド更新
- 回答生成は既定でGemini Flash-Liteを使用。回答待機上限は16秒
- 相談には会議に基づく具体策を中心に、一般知識の補足を区別して会話調で回答
- Slackには回答と参照ファイル名を表示し、原文抜粋は表示しない

他の検索方式やOpenAIを使う場合は、各フォルダの`requirements.txt`にある追加依存関係を導入してください。
API料金は利用するプロジェクト・プランの設定に依存します。

## テスト

```powershell
cd "Hermes Agent型"
python -X utf8 -B -m unittest -q
```

テストは外部APIや会議原文を必要としません。実APIの速度検証は、資料とAPIキーを設定したうえで実行します。

```powershell
python -X utf8 -B evaluate_advice_speed.py --api
```
