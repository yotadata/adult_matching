# DMM API 大量取得戦略

レビューデータと対応する全DMM動画データの効率的取得戦略

---

## 📊 分析結果サマリー

### レビューデータ分析
- **期間**: 2006年2月14日 〜 2025年9月5日（19年間、7,143日）
- **総レビュー数**: 38,904件
- **ユニークコンテンツ**: 32,304件
- **年代別分布**: 2024-2025年が全体の63%を占める

### DMM API能力分析
- **総動画数**: 約50,000件
- **API制限**: 1秒1コール推奨
- **ソートオプション**: date, price, review, rank
- **推定取得時間**: 約8.3時間（500ページ×1秒間隔）

---

## 🎯 効率化戦略

### Strategy 1: Multi-Sort Approach（推奨）
**異なるソート方法を組み合わせて効率的に古いデータまで取得**

```javascript
const SORT_STRATEGIES = [
  { sort: 'date', pages: 200 },    // 最新200ページ（最近のデータ）
  { sort: 'price', pages: 150 },   // 価格順150ページ（古いデータ含む）
  { sort: 'review', pages: 100 },  // レビュー順100ページ（人気データ）
  { sort: 'rank', pages: 50 }      // ランク順50ページ（ランキングデータ）
];
```

**利点**:
- 複数の観点からデータを取得し、カバレッジを最大化
- 古いデータから最新データまで効率的に網羅
- 重複は既存IDキャッシュで高速回避

### Strategy 2: Memory-Optimized Caching
**メモリキャッシュによる高速重複チェック**

```javascript
class OptimizedDuplicateCheck {
  constructor() {
    this.existingIds = new Set();     // メモリ内重複チェック
    this.batchCache = new Map();      // バッチ内重複防止
  }
  
  isDuplicate(contentId) {
    return this.existingIds.has(contentId) || this.batchCache.has(contentId);
  }
}
```

### Strategy 3: Adaptive Rate Limiting
**動的レート制御で最大効率を実現**

```javascript
class AdaptiveRateControl {
  constructor() {
    this.baseDelay = 800;           // 基本待機時間
    this.errorCount = 0;            // 連続エラー数
    this.successCount = 0;          // 連続成功数
  }
  
  getNextDelay() {
    if (this.errorCount > 0) {
      return this.baseDelay * Math.pow(2, this.errorCount); // 指数バックオフ
    } else if (this.successCount > 10) {
      return Math.max(600, this.baseDelay - 100); // 高速化
    }
    return this.baseDelay;
  }
}
```

### Strategy 4: Progressive Resume System
**中断・再開機能で確実な大量処理**

```json
// dmm_bulk_sync_progress.json
{
  "strategies": [
    { "sort": "date", "currentPage": 45, "maxPages": 200, "completed": false },
    { "sort": "price", "currentPage": 1, "maxPages": 150, "completed": false },
    { "sort": "review", "currentPage": 1, "maxPages": 100, "completed": false },
    { "sort": "rank", "currentPage": 1, "maxPages": 50, "completed": false }
  ],
  "totalProgress": "9%",
  "lastSaveAt": "2025-09-07T10:30:00Z"
}
```

---

## 🚀 実装プラン

### Phase 1: 準備フェーズ（5分）
1. **既存データキャッシュ作成**
   ```bash
   node scripts/create_id_cache.js
   # 既存1,000件のIDをメモリキャッシュ化
   ```

2. **進捗管理システム初期化**
   ```bash
   node scripts/init_bulk_sync.js
   # 進捗ファイル・キャッシュファイル作成
   ```

### Phase 2: 大量取得フェーズ（8-10時間）
1. **Multi-Sort並列取得**
   ```bash
   node scripts/efficient_dmm_bulk_sync.js --strategy multi-sort
   ```

2. **リアルタイム監視**
   ```bash
   # 別ターミナルで進捗監視
   node scripts/monitor_bulk_sync.js
   ```

### Phase 3: 品質検証フェーズ（30分）
1. **データ品質分析**
   ```bash
   node scripts/analyze_dmm_data.js --comprehensive
   ```

2. **コンテンツIDマッチング検証**
   ```bash
   node scripts/verify_content_matching.js
   ```

---

## ⚡ パフォーマンス最適化

### メモリ使用量最適化
```javascript
const MEMORY_OPTIMIZATION = {
  idCacheLimit: 100000,        // ID キャッシュ上限
  batchSize: 100,              // バッチサイズ
  periodicGC: 50,              // GC実行間隔（ページ数）
  memoryCheckInterval: 25      // メモリ使用量チェック間隔
};
```

### ネットワーク最適化
```javascript
const NETWORK_OPTIMIZATION = {
  timeout: 30000,              // タイムアウト30秒
  maxRetries: 3,               // 最大再試行回数
  retryBackoff: [1000, 2000, 4000], // 再試行間隔
  concurrency: 1,              // 同時実行数（API制限）
  keepAlive: true              // HTTP Keep-Alive
};
```

### ディスク I/O 最適化
```javascript
const DISK_OPTIMIZATION = {
  progressSaveInterval: 10,    // 進捗保存間隔
  cacheSaveInterval: 100,      // キャッシュ保存間隔
  batchCommitSize: 50,         // データベースコミットサイズ
  asyncWrites: true            // 非同期書き込み
};
```

---

## 📊 予想成果

### データカバレッジ
- **総取得予想数**: 45,000-50,000件（DMM全動画の90%以上）
- **レビューマッチング率**: 85%以上（32,304件中27,000件以上）
- **データ完全性**: 主要フィールド95%以上完備

### 処理時間見積もり
```
Multi-Sort Strategy (推奨):
├── Sort by date (200 pages): 3.3時間
├── Sort by price (150 pages): 2.5時間  
├── Sort by review (100 pages): 1.7時間
└── Sort by rank (50 pages): 0.8時間
Total: 約8.3時間
```

### システムリソース
```
Memory Usage: 200-500MB (IDキャッシュ含む)
Disk Space: +5GB (動画メタデータ + 関連テーブル)
Network: 約1.8GB転送 (500ページ × 平均3.6MB/ページ)
```

---

## 🛡️ エラー対策・監視

### Critical Error Handling
1. **API制限エラー**: 指数バックオフで自動回復
2. **ネットワークエラー**: 最大3回再試行
3. **データベースエラー**: トランザクション回復
4. **メモリ不足**: 定期的なGC + キャッシュクリア

### 監視メトリクス
```javascript
const MONITORING_METRICS = {
  pagesPerSecond: target >= 0.016,      // 秒あたり処理ページ数
  errorRate: target < 0.01,             // エラー率1%未満
  memoryUsage: target < 500,            // メモリ500MB未満
  duplicateRate: target >= 0.7          // 重複率70%以上（効率指標）
};
```

### 自動アラート
```bash
# システム監視スクリプト
*/15 * * * * node scripts/health_check.js
# 15分ごとに進捗・エラー率をチェック
# 異常時はSlack/メール通知
```

---

## 🎯 実行コマンド

### 標準実行
```bash
# 効率的大量取得（推奨）
node scripts/efficient_dmm_bulk_sync.js

# 進捗監視（別ターミナル）
node scripts/monitor_sync_progress.js
```

### カスタム実行
```bash
# 特定ソート方式のみ
node scripts/efficient_dmm_bulk_sync.js --sort-strategy date --pages 100

# 高速モード（レート制限緩和）
node scripts/efficient_dmm_bulk_sync.js --fast-mode --delay 600

# デバッグモード
node scripts/efficient_dmm_bulk_sync.js --debug --verbose
```

### 緊急停止・再開
```bash
# 緊急停止
pkill -f "efficient_dmm_bulk_sync"

# 進捗保存確認
cat scripts/dmm_bulk_sync_progress.json

# 再開
node scripts/efficient_dmm_bulk_sync.js --resume
```

---

**戦略実装開始準備完了** ✅  
**推定完了時刻**: 現在時刻 + 8.3時間  
**期待成果**: 45,000-50,000件のDMM動画データ取得