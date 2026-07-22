# AI-Loop-Trade003

FxPro MT5 **デモ実注文** + 5 グループ 12 銘柄の単独／グループ内ペア + Maker→Checker→Validator 自律ループ。

## Phase 1 ゴール

| 項目 | 内容 |
|------|------|
| ブローカー | FxPro MT5 **デモ**のみ（`mt5.require_demo: true`） |
| 執行 | `trading.dry_run: false` → `mt5.order_send` 実注文（スマホアプリで建玉確認可） |
| 銘柄 | Group1–5 の 12 銘柄（単独 + 同一 Group 内ペア） |
| 時間足 | M30 |
| ループ | 起動前 optimize → 常駐バー処理 → 週末レビュー |

## Requirements

- Windows（MT5 Python API）
- Python 3.11+
- MetaTrader 5（FxPro demo）常時ログイン
- `pip install -r requirements.txt`

## Quick Start（VPS）

1. FxPro デモで MT5 を起動しログインしたままにする
2. `config/settings.local.yaml` にログイン情報（コミットしない）:

```yaml
mt5:
  path: "C:/Program Files/FxPro - MetaTrader 5/terminal64.exe"  # 環境に合わせて
  login: 12345678
  password: "YOUR_DEMO_PASSWORD"
  server: "FxPro-MT5Demo"
```

3. 依存関係とデータ同期:

```bash
pip install -r requirements.txt
python main.py sync
python main.py status
```

4. 起動前パラメータ探索（単独 + ペア）:

```bash
python main.py optimize --pairs
```

5. 常駐ループ（デモ実注文）:

```bash
python main.py loop
```

または Task Scheduler / NSSM から:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_resident.ps1
```

6. MT5 スマホアプリで同一デモ口座の **建玉 / 履歴** を確認。ログの `ticket=` と突合する。

## Asset Groups

| Group | Strategy | Symbols |
|-------|----------|---------|
| Group1 | momentum_breakout | #US30, #USSPX500, #USNDAQ100, #Japan225 |
| Group2 | hybrid | #Germany40, #UK100 |
| Group3 | breakout_high_vol | GOLD, SILVER |
| Group4 | mean_reversion | EURUSD, GBPUSD, USDJPY |
| Group5 | breakout (×0.5 lots) | WTI |

ペアは Group 内のみ。ペア強度が閾値以上ならペア優先、さもなければ単独。同一銘柄の同時建玉は禁止。

ループエンジニアリング（起動前 optimize / 週末 review）は `check_all_asset_groups: true` により、取引サブセットに関係なく **全 Asset Group の銘柄＋グループ内ペアを毎回チェック**する。

`asset_rotation` により、複数 Group の同時適用と、強いシグナル側への緩やかな資金移行（migration）の余地を持つ。完全放棄はせず `min_group_weight` で復帰余地を残す。

## Safety

- デモ以外の口座ログインは起動時に拒否
- スプレッド > 0.03% はロット 0（Execution Guard）
- RiskAgent 拒否権・ドローダウン・サーキットブレーカ
- `graduation_stage: demo_live` は `dry_run=false` + `account_type=demo` 必須

## CLI

```bash
python main.py sync
python main.py run --symbol GOLD
python main.py optimize --pairs
python main.py review
python main.py loop
python main.py status
```

## Tests

```bash
python -m pytest tests/ -q
```
