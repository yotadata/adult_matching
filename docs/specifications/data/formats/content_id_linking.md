# Content ID リンキング仕様書

スクレイピングレビューデータとAPI取得動画データ間のContent ID紐づけ仕様

---

## 📋 概要

### 目的
- レビューデータのcontent_idとAPI動画データのexternal_idを紐づけ
- 疑似ユーザー生成時にAPI動画データの存在を確認
- ML学習時にはAPI動画データのみを使用することを保証

### データポリシー
```
✅ API動画データ (videos table) → メイン動画データソース
❌ レビュー動画メタデータ → 使用禁止
🔗 Content ID → 両データの橋渡し役のみ
```

---

## 🔗 Content ID マッピング

### データ構造
```python
ContentIDMapping = {
    "review_content_id": "dmm_12345",      # レビューデータのcontent_id
    "api_external_id": "12345",            # API videosテーブルのexternal_id  
    "match_confidence": 0.95,              # マッチング信頼度
    "match_method": "exact_id_match",      # マッチング手法
    "api_video_exists": True,              # API動画データ存在フラグ
    "created_at": "2025-09-05T12:00:00Z"
}
```

### マッチング戦略
```python
MATCHING_STRATEGIES = {
    # 優先度1: 完全一致
    "exact_id_match": {
        "method": lambda review_id, api_id: review_id == api_id,
        "confidence": 1.0
    },
    
    # 優先度2: プレフィックス除去マッチ
    "prefix_normalized_match": {
        "method": lambda review_id, api_id: 
                 review_id.replace('dmm_', '') == api_id,
        "confidence": 0.9
    },
    
    # 優先度3: ファジーマッチング
    "fuzzy_match": {
        "method": lambda review_id, api_id: 
                 fuzz.ratio(review_id, api_id) > 85,
        "confidence": 0.8
    }
}
```

---

## 🔄 リンキングプロセス

### Step 1: API動画データ準備
```python
def load_api_videos(supabase_client):
    """PostgreSQL videosテーブルから全動画取得"""
    response = supabase_client.table('videos').select('external_id, title, genre, maker').execute()
    return {video['external_id']: video for video in response.data}
```

### Step 2: レビューデータContent ID抽出
```python  
def extract_review_content_ids(review_data):
    """レビューデータからContent ID一覧取得"""
    content_ids = set()
    for review in review_data:
        if 'content_id' in review:
            content_ids.add(review['content_id'])
    return content_ids
```

### Step 3: Content ID マッピング実行
```python
def create_content_id_mapping(review_content_ids, api_videos):
    """Content ID マッピング作成"""
    mappings = []
    
    for review_id in review_content_ids:
        for strategy_name, strategy in MATCHING_STRATEGIES.items():
            for api_id in api_videos.keys():
                if strategy['method'](review_id, api_id):
                    mappings.append({
                        'review_content_id': review_id,
                        'api_external_id': api_id,
                        'match_method': strategy_name,
                        'match_confidence': strategy['confidence'],
                        'api_video_exists': True
                    })
                    break
    
    return mappings
```

### Step 4: 未マッチングContent ID処理
```python
def handle_unmatched_content_ids(review_content_ids, mappings):
    """API動画データに存在しないContent IDの処理"""
    matched_ids = {m['review_content_id'] for m in mappings}
    unmatched_ids = set(review_content_ids) - matched_ids
    
    # 未マッチングIDはML学習から除外
    excluded_mappings = []
    for unmatched_id in unmatched_ids:
        excluded_mappings.append({
            'review_content_id': unmatched_id,
            'api_external_id': None,
            'match_method': None,
            'match_confidence': 0.0,
            'api_video_exists': False,
            'excluded_from_training': True
        })
    
    return excluded_mappings
```

---

## 🎯 疑似ユーザー生成時の適用

### フィルタリング処理
```python
class ContentIDFilteredUserGenerator:
    def __init__(self, content_id_mappings):
        self.valid_content_ids = {
            mapping['review_content_id'] 
            for mapping in content_id_mappings 
            if mapping['api_video_exists']
        }
    
    def generate_pseudo_users(self, reviews):
        """API動画データが存在するレビューのみで疑似ユーザー生成"""
        filtered_reviews = []
        
        for review in reviews:
            if review['content_id'] in self.valid_content_ids:
                # API動画データ存在確認済みのレビューのみ使用
                filtered_reviews.append(review)
        
        return super().generate_pseudo_users(filtered_reviews)
```

### 統計レポート
```python
def generate_linking_report(mappings, total_reviews):
    """Content IDリンキング統計レポート"""
    return {
        'total_review_content_ids': len(set(m['review_content_id'] for m in mappings)),
        'successfully_linked': len([m for m in mappings if m['api_video_exists']]),
        'linking_success_rate': len([m for m in mappings if m['api_video_exists']]) / len(mappings),
        'excluded_from_training': len([m for m in mappings if not m['api_video_exists']]),
        'confidence_distribution': {
            'high (>0.9)': len([m for m in mappings if m['match_confidence'] > 0.9]),
            'medium (0.8-0.9)': len([m for m in mappings if 0.8 <= m['match_confidence'] <= 0.9]),
            'low (<0.8)': len([m for m in mappings if m['match_confidence'] < 0.8])
        }
    }
```

---

## 🔍 品質管理

### マッチング品質チェック
```python
QUALITY_THRESHOLDS = {
    'min_linking_success_rate': 0.70,      # 最低70%のContent IDがリンク成功
    'min_high_confidence_rate': 0.80,      # 80%以上が高信頼度マッチング
    'max_excluded_rate': 0.30               # 最大30%までの除外を許容
}

def validate_linking_quality(mappings):
    """リンキング品質の検証"""
    report = generate_linking_report(mappings)
    
    checks = {
        'linking_success_rate': report['linking_success_rate'] >= QUALITY_THRESHOLDS['min_linking_success_rate'],
        'high_confidence_rate': (report['confidence_distribution']['high (>0.9)'] / len(mappings)) >= QUALITY_THRESHOLDS['min_high_confidence_rate'],
        'excluded_rate': (report['excluded_from_training'] / len(mappings)) <= QUALITY_THRESHOLDS['max_excluded_rate']
    }
    
    return all(checks.values()), checks
```

### 不整合検出
```python
def detect_mapping_inconsistencies(mappings):
    """Content IDマッピングの不整合検出"""
    issues = []
    
    # 重複マッピング検出
    review_ids = [m['review_content_id'] for m in mappings]
    if len(review_ids) != len(set(review_ids)):
        issues.append("Duplicate review content IDs found")
    
    # 信頼度とマッチング手法の整合性
    for mapping in mappings:
        if mapping['match_method'] == 'exact_id_match' and mapping['match_confidence'] < 1.0:
            issues.append(f"Inconsistent confidence for exact match: {mapping['review_content_id']}")
    
    return issues
```

---

## 📊 実装例

### 完全なリンキングパイプライン
```python
def execute_content_id_linking_pipeline():
    """Content IDリンキング完全パイプライン"""
    
    # Step 1: データ準備
    supabase_client = create_supabase_client()
    api_videos = load_api_videos(supabase_client)
    review_data = load_scraped_reviews()
    review_content_ids = extract_review_content_ids(review_data)
    
    # Step 2: マッピング実行
    mappings = create_content_id_mapping(review_content_ids, api_videos)
    unmatched_mappings = handle_unmatched_content_ids(review_content_ids, mappings)
    all_mappings = mappings + unmatched_mappings
    
    # Step 3: 品質検証
    is_valid, quality_checks = validate_linking_quality(all_mappings)
    if not is_valid:
        raise ValueError(f"Content ID linking quality failed: {quality_checks}")
    
    # Step 4: 結果保存
    save_content_id_mappings(all_mappings)
    
    # Step 5: レポート生成
    report = generate_linking_report(all_mappings)
    return all_mappings, report
```

---

**文書管理**  
**最終更新**: 2025年9月5日  
**管理者**: Claude Code