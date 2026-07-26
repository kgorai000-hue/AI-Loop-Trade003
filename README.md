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

Anthropic API キーが無い／無効な場合は **settings.yaml のデフォルトを seed** してすぐ終わります（数分）。有効なキーがあるときだけ Maker→Checker→Validator→grid が走ります。

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

| Group | strategy（メタ） | Symbols |
|-------|------------------|---------|
| Group1 | momentum_breakout | #US30, #USSPX500, #USNDAQ100, #Japan225 |
| Group2 | hybrid | #Germany40, #UK100 |
| Group3 | breakout_high_vol | GOLD, SILVER |
| Group4 | mean_reversion | EURUSD, GBPUSD, USDJPY |
| Group5 | breakout (×0.5 lots) | WTI |

**Group.strategy は単独ルーティングに使いません**（ペア宇宙・ロット倍率・メタ情報用）。単独建玉は **5 状態レジーム**が戦略を決めます。

ペアは Group 内のみ。**関係性レジーム R1–R5（半減期・βドリフト・スプレッド拡散・片足ストレス）**が新規可否を決める。単独5状態は参考注記のみ（同一ファミリー必須ではない）。ペア強度が閾値以上ならペア優先。同一銘柄の同時建玉は禁止。

スプレッド定義: `S = log A − β log B`（βはローリングOLS）。R1通常 / R2閾値拡大・サイズ0.5 / R3–R5新規停止。

## レジーム（5 状態ハードカットオーバー）

| 状態 | 単独 | サイズ |
|------|------|--------|
| A `stable_trend` | trend_following | 100% |
| B `high_vol_trend` | trend_following | `high_vol_trend_scale`（既定 0.5） |
| C `stable_range` | mean_reversion | 100% |
| D `high_vol_chop` | 新規停止 | 0 |
| E `stress` | 新規停止 | 0 |
| `uncertain` | 新規停止 | 0 |

入力: 回帰傾き t 値、Efficiency Ratio、銘柄内 vol パーセンタイル、ATR ストレス代理、ベンチ相関。ヒステリシス＋`confirm_bars` で切替を抑制。

## optimize の検証範囲

`python main.py optimize`（および `loop` / `review` 内の optimize）は、**オフライン・バックテスト検証のみ**です。デモ実注文の実績は合格判定に使いません。

**002 / 旧 state は破棄してから実行してください**（戦略別パスに移行したため）。

```powershell
# 例: 旧 state を退避してクリア
Move-Item state state_archive_pre_five_state -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Path state | Out-Null
python main.py optimize --pairs
```

### パイプライン

```
Maker（候補提案） → Checker（敵対的却下） → Validator（数値ゲート） → 採用時のみ state/ 更新
```

| 段 | 役割 | 備考 |
|----|------|------|
| Maker | AppConfig 許可空間内のパラメータ候補 | `intelligence.maker_model`（既定 Sonnet） |
| Checker | 過適合・攻撃的閾値などを却下 | `intelligence.checker_model`（既定 Opus）。API キー無し時は grid fallback |
| Validator | バックテスト数値で Tier 判定 | LLM ではなく `BacktestAgent` + `loop_criteria` |

### 1 回の検証単位

| 軸 | 範囲 |
|----|------|
| 銘柄 | **1 銘柄ずつ**（`--pairs` 時はペア状態キー単位） |
| 戦略 | 既定で **trend_following と mean_reversion の両方**（`--strategy` で上書き可） |
| レジーム条件付き BT | シグナルを許可レジームのバーだけにマスク（trend→A/B、MR→C） |
| 時間足 | 既定 **M30** |
| state パス | `state/<Symbol>__<strategy>/`（例: `GOLD__trend_following`） |
| データ | MT5 同期済み OHLCV（`data/` DB） |
| 探索対象 | **許可パラメータ空間のみ** |

### Validator が実行する検証

`BacktestAgent.validate_strategy` が次をまとめて評価します。

1. レジーム条件付きシグナル生成
2. 単体バックテスト（コスト込み）
3. OOS 分割（`backtest.oos_*`）
4. Walk-Forward
5. Monte Carlo
6. Quality Gate
7. パラメータ感度（trend）

その結果を baseline と比較し、`src/backtest/loop_criteria.py` の `evaluate_trial` で判定します。

### 採用条件（合格の定義）

| 判定 | 意味 | 採用 |
|------|------|------|
| `hard_stop` | 取引数不足・MDD過大・OOS崩壊・期待ライブ悪化・MC P5 等 | しない |
| `reject` | Tier A 未満（WF Sharpe・改善幅・MDD など） | しない |
| `tier_a` | Tier A のみ通過、Tier B（頑健性）未達 | **しない** |
| `adopt`（Tier B） | Tier A + MC / WF 正例率 / 感度安定 / ゲート通過数 | **する** → `state/<銘柄>/` |

閾値の単一の真実は [`config/settings.yaml`](config/settings.yaml) の次です。

- `backtest:` … 窓・OOS・MC・ゲート
- `loop_engineering:` … Hard stop / Tier A / Tier B（例: `min_wf_test_sharpe`, `hard_stop_mdd_pct`, `tier_b_mc_prob_positive`）

### 検証に含まないもの

- デモ／ライブの実注文 PnL・スリッページ実績
- ポートフォリオ全体・複数銘柄の同時パラメータ最適化
- 資料にある Purged CV / Embargo / Triple Barrier（未実装）
- Meta-Labeling 二次モデル（未実装）

### 関連コマンド

```bash
# 全 tradeable 銘柄を順に optimize（既定）
python main.py optimize

# グループ内ペア状態も対象
python main.py optimize --pairs

# 特定銘柄のみ
python main.py optimize --symbol GOLD --symbol SILVER
```

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
