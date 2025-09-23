# 大規模レビューデータ収集仕様書

## 概要

DMM年間ランキング上位レビュワー（50名）から大量のレビューデータを収集し、Two-Tower推薦モデルの学習データを構築する

## 収集戦略

### Phase 1: トップレビュワー抽出
**対象URL**: `https://review.dmm.co.jp/review-front/ranking/1year`

**収集データ**:
- レビュワー名
- レビュー数（年間・累計）
- 参考になった数（年間・累計）
- レビュワーID（URLから抽出）

**HTML構造分析結果**:
```html
<!-- レビュワーエントリ -->
<a class="css-g65o95" href="https://www.dmm.co.jp/review/-/list/reviewer/=/id=[numeric-id]/">
  <div class="css-1akdctv">順位番号</div>
  <div class="css-735aui">ユーザー名</div>
  <div class="css-17diqim css-hs6ji3">参考になった数</div>
  <div>レビュー投稿数</div>
</a>
```

**抽出アルゴリズム**:
1. ランキングページからTOP10、TOP50セクションを特定
2. `.css-g65o95` クラスのリンク要素を全て抽出
3. URLから `id=([0-9]+)` パターンでレビュワーIDを抽出
4. 上位50名に制限してリスト化

### Phase 2: 個別レビュー収集
**対象URL**: `https://review.dmm.co.jp/review-front/reviewer/list/[reviewer_id]?page=[page_num]`

**収集データ**:
```json
{
  "contentId": "商品ID",
  "displayShopName": "カテゴリ",
  "title": "レビュータイトル",
  "text": "レビュー本文",
  "value": "評価点(1-5)",
  "writeDate": "投稿日時",
  "contentTitle": "商品タイトル",
  "contentUrl": "商品URL",
  "contentImageSrc": "商品画像URL",
  "evaluateCount": "参考になった数",
  "reviewer_id": "レビュワーID",
  "page_number": "収集ページ番号"
}
```

**データソース構造**:
- レビューデータは `reviewList` JSON配列に格納
- 動的レンダリングされるため、JavaScriptの実行または直接API呼び出しが必要
- ページネーションで複数ページに分散

## 技術実装仕様

### 1. トップレビュワー抽出スクリプト

**ファイル**: `data_processing/scraping/top_reviewers_scraper.py`

**機能**:
- Cookie認証による年齢確認回避
- ランキングページのHTMLパース
- レビュワー情報の構造化データ抽出
- JSON形式での保存

**出力**: `data_processing/raw_data/top_reviewers.json`

```python
class TopReviewersScraper:
    def extract_reviewers(self) -> List[Dict]:
        """上位50名のレビュワー情報を抽出"""
        pass
    
    def parse_reviewer_element(self, element) -> Dict:
        """個別レビュワー要素から情報抽出"""
        pass
```

### 2. バッチレビュー収集スクリプト

**ファイル**: `data_processing/scraping/batch_review_scraper.py`

**機能**:
- 複数レビュワーの並列処理
- ページネーション自動追跡
- レート制限・エラーハンドリング
- 進捗追跡・中断再開機能

**出力**: `data_processing/raw_data/batch_reviews/reviewer_[id].json`

```python
class BatchReviewScraper:
    def collect_all_reviewers(self, reviewer_list: List[Dict]) -> None:
        """全レビュワーのデータを収集"""
        pass
    
    def collect_reviewer_reviews(self, reviewer_id: str) -> List[Dict]:
        """特定レビュワーの全レビューを収集"""
        pass
    
    def auto_paginate(self, reviewer_id: str) -> Iterator[List[Dict]]:
        """ページネーション自動処理"""
        pass
```

### 3. データ統合・前処理

**機能**:
- 複数ファイルの統合
- 重複データの除去
- データ品質チェック
- 統計情報生成

## 品質保証

### データ検証項目
- [ ] レビューテキストの文字数（最低10文字）
- [ ] 評価点の妥当性（1-5の範囲）
- [ ] 投稿日時の形式確認
- [ ] 商品情報の完整性
- [ ] レビュワーIDの一意性

### エラーハンドリング
- [ ] HTTP接続エラー
- [ ] Cookie認証失敗
- [ ] レート制限対応（3秒間隔）
- [ ] パースエラー処理
- [ ] 中断時の状態保存

### 期待収集量
- **目標レビュワー数**: 50名
- **期待レビュー数**: 100,000件以上
- **1レビュワーあたり平均**: 2,000件
- **推定実行時間**: 6-8時間

## セキュリティ・倫理

### 遵守事項
- [ ] robots.txtの確認・遵守
- [ ] レート制限の実装（サーバー負荷軽減）
- [ ] 個人情報の適切な取り扱い
- [ ] 利用規約の遵守
- [ ] 研究・開発目的での使用に限定

### データ保護
- [ ] 収集データの暗号化
- [ ] アクセス権限の制限
- [ ] 定期的なバックアップ
- [ ] データ保持期間の設定

## 実装スケジュール

### Phase 1: トップレビュワー抽出 (1日)
- [ ] スクリプト実装
- [ ] テスト・デバッグ
- [ ] 実データ収集

### Phase 2: バッチレビュー収集 (2日)
- [ ] バッチ処理実装
- [ ] エラーハンドリング強化
- [ ] 大規模データ収集実行

### Phase 3: データ統合・検証 (1日)
- [ ] データ統合処理
- [ ] 品質チェック実行
- [ ] 疑似ユーザー生成テスト

## 成功指標

### 定量指標
- レビュワー抽出成功率: 100%
- レビュー収集成功率: 95%以上
- データ品質スコア: 90%以上
- 処理エラー率: 5%以下

### 定性指標
- レビュー内容の多様性
- 商品カテゴリの網羅性
- 時間軸の分散度
- 評価分布の自然性

---

**作成日**: 2025年9月5日  
**更新日**: 2025年9月5日  
**作成者**: Claude Code  
**レビュー担当**: -