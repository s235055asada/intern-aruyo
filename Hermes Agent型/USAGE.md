# Hermes Agent 操作ガイド

この文書は、PowerShellでHermes Agentを起動し、会議ログへ質問して終了するまでの
日常操作を説明します。

## 1. 初回だけ行う準備

PowerShellを開き、プロジェクトへ移動します。

```powershell
cd "C:\aruyoインターン\Hermes Agent型"
```

必要なPythonパッケージが未導入の場合だけ実行します。

```powershell
python -m pip install -r .\requirements.txt
```

会議ログの保存先は `config.json` で `..\MD` に設定済みです。

## 2. APIキーを設定する

### 今開いているPowerShellだけで使う

```powershell
$env:GEMINI_API_KEY="あなたのGemini APIキー"
```

PowerShellを閉じると、この設定は消えます。

### Windowsユーザー環境変数へ登録する

```powershell
[Environment]::SetEnvironmentVariable(
  "GEMINI_API_KEY",
  "あなたのGemini APIキー",
  "User"
)
```

登録後はPowerShellを一度閉じ、新しく開いてください。登録済みかどうかは、キー本体を
表示せず次のコマンドで確認できます。

```powershell
if ([Environment]::GetEnvironmentVariable("GEMINI_API_KEY", "User")) {
  "GEMINI_API_KEY: 設定済み"
} else {
  "GEMINI_API_KEY: 未設定"
}
```

通常運用ではAPIキーをMarkdown、ソースコード、Git管理ファイルへ書かないでください。
現在のテスト構成に限り、環境変数が未設定なら`..\目標１.txt`末尾のGeminiキーを
参照します。

## 3. チャットを起動する

```powershell
cd "C:\aruyoインターン\Hermes Agent型"
python .\hermes_agent.py --chat --provider gemini
```

通常はローカルBM25の `keyword` 検索が自動的に使われます。GBrainを併用する場合は
`--search enhanced`を明示してください。

起動に成功すると次の表示になります。

```text
会議ログAIチャットを開始しました。
終了: /exit  履歴を消去: /clear  ヘルプ: /help

あなた>
```

## 4. 質問する

`あなた>` の後ろへ質問を入力し、Enterを押します。

```text
あなた> 重複した候補者データはどう処理する？
```

回答後も `あなた>` が表示されるため、そのまま次の質問を入力できます。

```text
あなた> その理由は？
あなた> 担当者は誰？
あなた> ほかに注意点はある？
```

直前の会話履歴を保持しているため、「その担当者」「その理由」のような続きの質問が
できます。別の話題へ移るときは、対象を具体的に書くか `/clear` を使ってください。

### 質問例

```text
オンボーディングで決まった手順を教えて
候補者メールはどの順番で対応する？
中途採用を継続できる人材の条件は？
スカウト文面について何が決まった？
AIへ機密情報を入力するときの注意点は？
```

一度に多くの話題を聞くより、1回の質問を1つの目的に絞る方が根拠を確認しやすくなります。

## 5. 回答を確認する

回答中の `[S1]`、`[S2]` は根拠IDです。

```text
候補者メールは古いものから対応します [S1]。

根拠:
- [S1] オンボ-3.md / chunk 12: （実際の原文抜粋）
```

確認するポイントは次の3点です。

1. 事実の直後に `[S番号]` が付いているか
2. 根拠欄のファイル名が質問と関係しているか
3. 原文抜粋が回答内容を直接示しているか

担当者や期限などがログにない場合は「確認できません」と表示されます。重要な判断では、
回答だけでなく根拠欄の原文も確認してください。

## 6. チャット内コマンド

| コマンド | 動作 |
| --- | --- |
| `/help` | 利用できるコマンドを表示する |
| `/clear` | 直前までの会話履歴を消す |
| `/exit` | チャットを終了する |

`Ctrl+C` でも終了できますが、通常は `/exit` を使ってください。

## 7. 1回だけ質問する

チャットを開かず、1問だけ回答して終了する場合:

```powershell
python .\hermes_agent.py "候補者メールはどの順番で対応する？" --provider gemini
```

検索結果だけを確認し、Geminiの回答を生成しない場合:

```powershell
python .\hermes_agent.py "候補者メールはどの順番で対応する？" `
  --provider gemini `
  --show-search-results
```

## 8. APIを使わない検索

検索部分でGBrain/Gemini Embeddingを使わず、ローカルBM25だけで検索する場合:

```powershell
python .\hermes_agent.py --chat --provider gemini --search keyword
```

この場合も最終回答の生成にはGemini APIを使用します。APIを一切呼ばずに確認できるのは、
`--search keyword --show-search-results` を指定した検索結果表示だけです。

## 9. 終了後にもう一度起動する

同じPowerShellで再起動する場合:

```powershell
python .\hermes_agent.py --chat --provider gemini
```

新しいPowerShellでは、APIキーを一時設定していた場合のみ再設定が必要です。Windowsの
ユーザー環境変数へ登録している場合は再設定不要です。

## 10. よくあるエラー

### APIキーが設定されていない

```text
GEMINI_API_KEY または GOOGLE_API_KEY が設定されていません。
```

今開いているPowerShellで次を実行し、再起動します。

```powershell
$env:GEMINI_API_KEY="あなたのGemini APIキー"
python .\hermes_agent.py --chat --provider gemini
```

### `python` が見つからない

```powershell
py .\hermes_agent.py --chat --provider gemini
```

でも起動できるか確認してください。どちらも動かない場合はPythonのインストールとPATH設定が
必要です。

### `GBrainが見つかりません` と表示される

GBrainの確認:

```powershell
& "$env:USERPROFILE\.bun\bin\gbrain.exe" --version
```

一時的にBM25だけで起動する場合:

```powershell
python .\hermes_agent.py --chat --provider gemini --search keyword
```

### Gemini APIで429が表示される

無料枠などのレート制限です。しばらく待ってから再実行してください。繰り返し発生する場合は、
Google AI Studioで対象プロジェクトの利用上限を確認してください。

### 回答が「確認できません」になる

次を順番に確認します。

1. 質問対象のMarkdownが `C:\aruyoインターン\MD` にあるか
2. 人名、会社名、案件名などを質問へ追加できるか
3. 検索結果だけを表示して、関連ファイルが取得されているか

```powershell
python .\hermes_agent.py "質問文" --provider gemini --show-search-results --top-k 8
```

関連ファイルが表示されても原文に答えがなければ、「確認できません」が正しい動作です。

## 11. 会議ログを追加した後

`C:\aruyoインターン\MD` へ新しいMarkdownを追加した場合、BM25検索には次回起動時から
自動的に反映されます。GBrainへ反映するには、プロジェクト管理者がGBrainの取り込みと
Embedding更新を行う必要があります。更新前は `--search keyword` なら新しいMDも検索できます。
