# Slack連携セットアップ

Hermes Meeting BotはSlack Socket Modeで動作します。Botを動かすPCからSlackへ
WebSocket接続するため、公開URL、固定IP、ポート開放は不要です。

## 1. Slack Appを作成する

1. [Slack APIのYour Apps](https://api.slack.com/apps)を開く。
2. `Create New App`、`From an app manifest`の順に選ぶ。
3. Botを追加するSlackワークスペースを選ぶ。
4. `YAML`を選び、`slack_manifest.yaml`の内容を貼り付ける。
5. 内容を確認してAppを作成する。

Manifestには次のBot権限が含まれています。

- `app_mentions:read`: チャンネル内のBotメンションを受信
- `chat:write`: スレッドへ回答を投稿
- `channels:history`: パブリックチャンネルのスレッド返信を受信
- `groups:history`: プライベートチャンネルのスレッド返信を受信

受信イベントは`app_mention`、`message.channels`、`message.groups`です。

## 2. App Tokenを作る

1. Slack App管理画面の`Basic Information`を開く。
2. `App-Level Tokens`の`Generate Token and Scopes`を選ぶ。
3. Token Nameへ`socket-mode`などを入力する。
4. `connections:write`を追加して生成する。
5. `xapp-`で始まる値を控える。

これは`SLACK_APP_TOKEN`として使用します。

## 3. ワークスペースへインストールする

1. `OAuth & Permissions`を開く。
2. `Install to Workspace`または`Reinstall to Workspace`を実行する。
3. `Bot User OAuth Token`に表示される`xoxb-`で始まる値を控える。

これは`SLACK_BOT_TOKEN`として使用します。トークンをMarkdown、ソースコード、
チャットへ貼り付けないでください。

## 4. Botをチャンネルへ追加する

今回の対象チャンネル:

- URL: `https://YOUR_WORKSPACE.slack.com/archives/YOUR_CHANNEL_ID`
- チャンネルID: `YOUR_CHANNEL_ID`
- チャンネルIDは利用環境で設定してください

用意したチャンネルを開き、次を投稿します。

```text
/invite @Hermes Meeting Bot
```

チャンネル名をクリックし、チャンネル詳細のリンクからチャンネルIDを確認します。
通常は`C`で始まる文字列です。チャンネルIDを設定すると、Botはそのチャンネル以外の
質問へ回答しません。

## 5. Windowsユーザー環境変数を設定する

PowerShellで次を実行します。値部分はSlack管理画面で取得した値へ置き換えます。

```powershell
[Environment]::SetEnvironmentVariable(
  "SLACK_BOT_TOKEN",
  "xoxb-ここにBot User OAuth Token",
  "User"
)

[Environment]::SetEnvironmentVariable(
  "SLACK_APP_TOKEN",
  "xapp-ここにApp-Level Token",
  "User"
)

[Environment]::SetEnvironmentVariable(
  "SLACK_CHANNEL_ID",
  "YOUR_CHANNEL_ID",
  "User"
)
```

値を表示せず、設定状態だけを確認できます。

```powershell
@("SLACK_BOT_TOKEN", "SLACK_APP_TOKEN", "SLACK_CHANNEL_ID") | ForEach-Object {
  if ([Environment]::GetEnvironmentVariable($_, "User")) {
    "${_}: 設定済み"
  } else {
    "${_}: 未設定"
  }
}
```

GeminiキーはCLI版と共通です。現在のテスト構成では環境変数がなければ
`..\目標１.txt`末尾のキーを参照します。

## 6. Botを起動する

```powershell
cd "C:\aruyoインターン\Hermes Agent型"
python -m pip install -r .\requirements.txt
python .\slack_bot.py
```

次の表示後はPowerShellを閉じずに起動したままにします。

```text
Hermes Slack Botを起動します。対象: Cから始まるチャンネルID
```

既定では`gemini-2.5-flash-lite`とローカルBM25検索を使います。検索方式などを変える場合は次のように
指定できます。

```powershell
python .\slack_bot.py --provider gemini --search keyword --top-k 6
```

停止するときはPowerShellで`Ctrl+C`を押します。

## 7. Slackで質問する

チャンネルで最初の質問にBotをメンションします。

```text
@Hermes Meeting Bot オンボーディングについて決まったことを教えて
```

回答は質問のスレッドへ投稿されます。同じスレッド内では、2回目以降はメンションせず
続けて質問できます。

```text
その担当者は誰？
期限は？
```

スレッド内で`/clear`と投稿すると、そのスレッドの会話履歴を削除します。会話履歴は
Botプロセスのメモリ内だけに保存されるため、Botを再起動すると消えます。

## 動作上の注意

- 資料と検索用情報は起動時にメモリへ読み込みます。通常の質問ではDriveへアクセスしません。
- 前回の取得から24時間経過後の質問をきっかけに、バックグラウンドで資料を更新します。
  更新中の質問には前回の資料を使い、更新失敗時も前回の資料を維持します。即時反映はBotを再起動してください。
- 異なるスレッドは最大3件まで並行処理します。同じスレッドへの連続投稿や上限超過時は、待ち行列を増やさず処理中と案内します。
- 回答生成の待機上限は16秒です。上限を超えた場合は時間内に完了しなかった旨を返します。
  通信障害やSlack側の遅延を含む厳密な20秒以内の配送保証ではありません。
- 「経理採用を上手く行かせるには」「エンジニアの選考フローどうしたら」などの相談は、
  職種に関係する根拠を検索し、「提案」と確認質問を返します。提案は会議の決定事項を意味しません。
- 相談回答は700〜1000字を目安に、会議での背景と具体的な提案を中心（約7割）にし、
  モデルの一般知識による補足（約3割）を分けて表示します。この比率は内容構成の目安です。
  一般知識には会議の引用を付けず、原文にない会社の実績・担当者・期限を補いません。
- 相談には今回の進め方を先に勧め、過去の会議で何を比較・重視して検討したかを
  理由として説明します。過去の決定事項の一覧ではなく、過去の考え方を今回へ応用する形式です。
  原文に判断経緯や実績がなければ、決定済み・成功事例とは表現しません。
- 長い回答はSlackの上限に合わせて複数メッセージへ分割します。
- `SLACK_CHANNEL_ID`を設定しない場合、Botが参加している全チャンネルが対象です。
- PCがスリープ、シャットダウン、またはBot停止中の場合は回答できません。
- 本番で常時利用する場合は、常時稼働する社内PCまたはサーバーで実行してください。

## よくあるエラー

### `SLACK_APP_TOKENが設定されていません`

`xapp-`で始まるApp-Level TokenをWindowsユーザー環境変数へ登録します。

### `SLACK_BOT_TOKENが設定されていません`

Appをワークスペースへインストールし、`xoxb-`で始まるBot Tokenを登録します。

### メンションしても反応しない

次を確認します。

1. PowerShellで`slack_bot.py`が起動中か。
2. Slack AppでSocket Modeが有効か。
3. App Tokenに`connections:write`があるか。
4. Botが対象チャンネルへ招待されているか。
5. `SLACK_CHANNEL_ID`が実際のチャンネルIDと一致しているか。
6. 権限変更後に`Reinstall to Workspace`を実行したか。

## 公式資料

- [Bolt for PythonでSlack Appを作成](https://docs.slack.dev/tools/bolt-python/creating-an-app/)
- [Socket Modeの利用方法](https://docs.slack.dev/tools/bolt-python/concepts/socket-mode/)
