# レビューデータ学習仕様書

**プロジェクト**: アダルト動画マッチングアプリケーション  
**文書バージョン**: v1.0  
**作成日**: 2025年9月3日  
**目的**: レビューデータを疑似ユーザーデータとして活用した初期学習システムの設計

---

## 📋 目次

1. [課題と解決アプローチ](#課題と解決アプローチ)
2. [レビューデータ分析](#レビューデータ分析) 
3. [疑似ユーザー生成戦略](#疑似ユーザー生成戦略)
4. [学習データ変換設計](#学習データ変換設計)
5. [ML学習パイプライン](#ml学習パイプライン)
6. [品質保証・検証](#品質保証検証)
7. [実装計画](#実装計画)

---

## 🎯 課題と解決アプローチ

### 現状の課題
**冷開始問題 (Cold Start Problem)**
- 新しいアプリケーションにはユーザー行動データが存在しない
- Two-Towerモデルの学習には大量のユーザー-アイテム相互作用が必要
- 実際のユーザー行動データ収集には数ヶ月〜数年が必要

### 解決アプローチ
**レビューデータからの疑似ユーザー学習**
1. **レビューデータ → 疑似ユーザー変換**: レビュアーを疑似ユーザーとして扱い
2. **評価値 → いいね行動変換**: 評価スコアを Like/Skip 行動に変換
3. **テキスト → ユーザー嗜好変換**: レビューテキストからユーザーの好みを抽出
4. **学習済みモデル → 実ユーザー適応**: 実ユーザーデータでファインチューニング

### 期待効果
- ✅ **即座の運用開始**: アプリリリース時点で推薦機能提供
- ✅ **多様な嗜好学習**: 様々なレビュアーの好みパターンを学習
- ✅ **品質の事前検証**: 実ユーザー前にモデル品質を検証
- ✅ **段階的改善**: 実データで継続的に精度向上

---

## 📊 レビューデータ分析

### 現在取得済みデータ
```json
{
  "total_reviews": 21,
  "valid_after_cleaning": 3,
  "data_structure": {
    "review_text": "詳細なレビューテキスト",
    "rating": 1.0,  // 数値評価（一部データ）
    "element_info": {...}  // HTML構造情報
  }
}
```

### レビューデータの特徴分析

#### データ品質評価
```python
# data_processing/analysis/review_analysis.py

class ReviewDataAnalyzer:
    def analyze_review_quality(self, reviews):
        return {
            'text_length_distribution': self.analyze_text_lengths(reviews),
            'sentiment_distribution': self.analyze_sentiments(reviews), 
            'rating_distribution': self.analyze_ratings(reviews),
            'reviewer_diversity': self.analyze_reviewer_patterns(reviews),
            'content_coverage': self.analyze_content_coverage(reviews)
        }
```

#### 拡張データ収集要件
```yaml
target_data_volume:
  minimum_reviews: 100000  # 最低10万件
  minimum_videos: 10000    # 最低1万動画
  minimum_reviewers: 5000  # 最低5千レビュアー

data_quality_requirements:
  text_length_min: 20      # 20文字以上のレビュー
  text_length_max: 5000    # 5000文字以下
  rating_coverage: 0.3     # 30%以上に評価値あり
  reviewer_min_reviews: 3  # レビュアーあたり最低3件

data_diversity_requirements:
  genre_coverage: 0.8      # 80%以上のジャンルカバー
  rating_variance: 1.5     # 評価値の分散1.5以上
  temporal_span: 12months  # 12ヶ月間のデータ
```

---

## 👤 疑似ユーザー生成戦略

### レビュアー → 疑似ユーザー変換

#### 1. レビュアーID抽出・匿名化
```python
class PseudoUserGenerator:
    def extract_reviewers(self, reviews):
        """レビューデータからレビュアーIDを抽出"""
        reviewers = {}
        for review in reviews:
            reviewer_id = review.get('external_reviewer_id') or \
                         self.generate_pseudo_reviewer_id(review)
            
            if reviewer_id not in reviewers:
                reviewers[reviewer_id] = {
                    'pseudo_user_id': self.anonymize_reviewer_id(reviewer_id),
                    'reviews': [],
                    'profile': self.init_user_profile()
                }
            
            reviewers[reviewer_id]['reviews'].append(review)
        
        return reviewers
    
    def anonymize_reviewer_id(self, reviewer_id):
        """レビュアーIDを匿名化してユーザーIDに変換"""
        return hashlib.sha256(f"pseudo_{reviewer_id}_{self.salt}".encode()).hexdigest()
```

#### 2. ユーザープロファイル生成
```python
def generate_user_profile(self, reviewer_reviews):
    """レビュー履歴からユーザープロファイル生成"""
    profile = {
        'user_id': reviewer['pseudo_user_id'],
        'review_count': len(reviewer_reviews),
        'avg_rating': np.mean([r.get('rating', 0) for r in reviewer_reviews]),
        'rating_variance': np.var([r.get('rating', 0) for r in reviewer_reviews]),
        'review_length_avg': np.mean([len(r.get('review_text', '')) for r in reviewer_reviews]),
        'sentiment_profile': self.analyze_sentiment_profile(reviewer_reviews),
        'genre_preferences': self.extract_genre_preferences(reviewer_reviews),
        'temporal_activity': self.analyze_temporal_activity(reviewer_reviews)
    }
    return profile
```

### ユーザー行動データ生成

#### 評価値 → Like/Skip 変換ロジック
```python
class ReviewToActionConverter:
    def __init__(self):
        # 評価値からいいね確率への変換テーブル
        self.rating_to_like_probability = {
            5.0: 0.95,  # 5つ星 → 95%いいね
            4.0: 0.80,  # 4つ星 → 80%いいね  
            3.0: 0.50,  # 3つ星 → 50%いいね
            2.0: 0.20,  # 2つ星 → 20%いいね
            1.0: 0.05   # 1つ星 → 5%いいね
        }
    
    def convert_review_to_actions(self, review, video_id, user_id):
        """レビュー1件をユーザー行動データに変換"""
        actions = []
        
        # 1. メインアクション（レビューしたということ＝視聴した）
        main_action = self.generate_main_action(review, video_id, user_id)
        actions.append(main_action)
        
        # 2. 関連動画への推定アクション（同じ出演者・ジャンル等）
        related_actions = self.generate_related_actions(review, user_id)
        actions.extend(related_actions)
        
        return actions
    
    def generate_main_action(self, review, video_id, user_id):
        """メインアクション生成"""
        rating = review.get('rating', 3.0)
        sentiment_score = self.analyze_sentiment(review.get('review_text', ''))
        
        # 評価値と感情スコアを組み合わせていいね確率を決定
        like_prob = self.calculate_like_probability(rating, sentiment_score)
        action = 'like' if random.random() < like_prob else 'skip'
        
        return {
            'user_id': user_id,
            'video_id': video_id,
            'action': action,
            'confidence': like_prob if action == 'like' else (1 - like_prob),
            'timestamp': self.parse_review_timestamp(review),
            'source': 'review_conversion',
            'metadata': {
                'original_rating': rating,
                'sentiment_score': sentiment_score,
                'review_length': len(review.get('review_text', ''))
            }
        }
```

#### 関連動画への推定行動生成
```python
def generate_related_actions(self, review, user_id, num_actions=5):
    """レビューから関連動画への行動を推定生成"""
    
    # レビューテキストからキーワード抽出
    keywords = self.extract_keywords(review['review_text'])
    preferences = self.infer_preferences(review)
    
    # 類似動画を検索
    similar_videos = self.find_similar_videos(keywords, preferences)
    
    actions = []
    for video in similar_videos[:num_actions]:
        # 好み度に基づいてアクション確率を計算
        preference_score = self.calculate_preference_match(preferences, video)
        action_prob = self.preference_to_action_probability(preference_score)
        
        action = {
            'user_id': user_id,
            'video_id': video['id'],
            'action': 'like' if random.random() < action_prob else 'skip',
            'confidence': action_prob,
            'timestamp': self.generate_realistic_timestamp(review),
            'source': 'inferred_preference',
            'metadata': {
                'preference_score': preference_score,
                'similarity_basis': keywords,
                'inference_confidence': 0.6  # 推定データなので低い信頼度
            }
        }
        actions.append(action)
    
    return actions
```

---

## 🔄 学習データ変換設計

### データ変換パイプライン

```python
# ml_pipeline/preprocessing/review_to_training_data.py

class ReviewToTrainingDataPipeline:
    def __init__(self):
        self.user_generator = PseudoUserGenerator()
        self.action_converter = ReviewToActionConverter()
        self.feature_extractor = UserFeatureExtractor()
        self.data_augmenter = DataAugmenter()
    
    def process_reviews_to_training_data(self, reviews):
        """レビューデータを訓練データに全体変換"""
        
        # Step 1: 疑似ユーザー生成
        pseudo_users = self.user_generator.extract_reviewers(reviews)
        print(f"Generated {len(pseudo_users)} pseudo users")
        
        # Step 2: ユーザー行動データ変換
        user_actions = []
        for user_id, user_data in pseudo_users.items():
            actions = self.action_converter.convert_user_reviews(user_data)
            user_actions.extend(actions)
        
        print(f"Generated {len(user_actions)} user actions")
        
        # Step 3: 特徴量エンジニアリング
        user_features = self.feature_extractor.extract_user_features(pseudo_users)
        item_features = self.feature_extractor.extract_item_features(reviews)
        
        # Step 4: データ拡張・バランス調整
        augmented_data = self.data_augmenter.augment_training_data(
            user_actions, user_features, item_features
        )
        
        # Step 5: 訓練用フォーマット変換
        training_dataset = self.format_for_training(augmented_data)
        
        return training_dataset
```

### 特徴量設計

#### ユーザー特徴量（疑似ユーザーベース）
```python
class UserFeatureExtractor:
    def extract_user_features(self, pseudo_users):
        features = {}
        
        for user_id, user_data in pseudo_users.items():
            reviews = user_data['reviews']
            
            # 基本統計特徴量
            basic_features = {
                'review_count': len(reviews),
                'avg_rating': np.mean([r.get('rating', 3.0) for r in reviews]),
                'rating_std': np.std([r.get('rating', 3.0) for r in reviews]),
                'avg_review_length': np.mean([len(r.get('review_text', '')) for r in reviews])
            }
            
            # テキスト分析特徴量
            text_features = self.extract_text_features(reviews)
            
            # 嗜好特徴量
            preference_features = self.extract_preference_features(reviews)
            
            # 行動パターン特徴量
            behavioral_features = self.extract_behavioral_features(reviews)
            
            features[user_id] = {
                **basic_features,
                **text_features,
                **preference_features,
                **behavioral_features
            }
        
        return features
    
    def extract_preference_features(self, reviews):
        """レビューから嗜好特徴量を抽出"""
        
        # ジャンル嗜好
        genre_mentions = Counter()
        performer_mentions = Counter()
        sentiment_scores = []
        
        for review in reviews:
            text = review.get('review_text', '')
            
            # ジャンル・出演者への言及を抽出
            genres = self.extract_genre_mentions(text)
            performers = self.extract_performer_mentions(text)
            
            genre_mentions.update(genres)
            performer_mentions.update(performers)
            
            # 感情分析
            sentiment = self.analyze_sentiment(text)
            sentiment_scores.append(sentiment)
        
        return {
            'preferred_genres': dict(genre_mentions.most_common(5)),
            'preferred_performers': dict(performer_mentions.most_common(3)),
            'avg_sentiment': np.mean(sentiment_scores),
            'sentiment_variance': np.var(sentiment_scores),
            'positive_review_ratio': sum(1 for s in sentiment_scores if s > 0.1) / len(sentiment_scores)
        }
```

### データ拡張戦略

#### 1. 負例生成（ネガティブサンプリング）
```python
class NegativeSampler:
    def generate_negative_samples(self, positive_actions, ratio=2.0):
        """ポジティブアクション（いいね）に対してネガティブ（スキップ）を生成"""
        
        negative_actions = []
        
        for pos_action in positive_actions:
            if pos_action['action'] == 'like':
                # 同ユーザーの嫌いそうな動画を推定
                negative_videos = self.find_dissimilar_videos(
                    user_id=pos_action['user_id'],
                    liked_video_id=pos_action['video_id'],
                    count=int(ratio)
                )
                
                for video_id in negative_videos:
                    neg_action = {
                        **pos_action,
                        'video_id': video_id,
                        'action': 'skip',
                        'confidence': 0.8,  # 推定なので信頼度は低め
                        'source': 'negative_sampling'
                    }
                    negative_actions.append(neg_action)
        
        return negative_actions
```

#### 2. 時系列データ生成
```python
class TemporalDataGenerator:
    def generate_temporal_patterns(self, user_actions):
        """リアルな時系列アクセスパターンを生成"""
        
        # ユーザーごとにセッションを生成
        sessions = {}
        for user_id in set(action['user_id'] for action in user_actions):
            user_actions_sorted = sorted([a for a in user_actions if a['user_id'] == user_id], 
                                       key=lambda x: x['timestamp'])
            
            # セッション分割（1時間以上空いたら新セッション）
            sessions[user_id] = self.split_into_sessions(user_actions_sorted)
        
        return sessions
```

---

## 🤖 ML学習パイプライン

### Two-Tower モデル適応

#### 疑似ユーザー学習用の修正
```python
# ml_pipeline/models/two_tower_pseudo_user.py

class TwoTowerPseudoUserModel(TwoTowerModel):
    def __init__(self, config):
        super().__init__(config)
        
        # 疑似ユーザー用の追加レイヤー
        self.confidence_weight_layer = tf.keras.layers.Dense(1, activation='sigmoid')
        self.source_embedding = tf.keras.layers.Embedding(
            config.source_vocab_size, 32  # review_conversion, inferred_preference, negative_sampling
        )
    
    def call(self, inputs):
        user_features, item_features, confidence, source = inputs
        
        # 基本的なTwo-Tower処理
        user_repr = self.user_tower(user_features)
        item_repr = self.item_tower(item_features)
        
        # 疑似データの信頼度を考慮
        confidence_weight = self.confidence_weight_layer(confidence)
        source_emb = self.source_embedding(source)
        
        # 類似度計算
        similarity = tf.keras.utils.cosine_similarity(user_repr, item_repr, axis=1)
        
        # 信頼度で重み付け
        weighted_similarity = similarity * confidence_weight
        
        # 最終予測
        prediction = tf.nn.sigmoid(weighted_similarity + 
                                 tf.reduce_mean(source_emb, axis=1))
        
        return prediction
```

#### 段階的学習戦略
```python
class StageWiseLearning:
    def __init__(self, model):
        self.model = model
        self.learning_stages = [
            'pseudo_user_pretraining',
            'real_user_adaptation', 
            'continuous_learning'
        ]
    
    def stage1_pseudo_user_pretraining(self, pseudo_training_data):
        """段階1: 疑似ユーザーデータでの事前学習"""
        
        print("Stage 1: Pseudo User Pre-training")
        
        # 疑似データに特化した学習設定
        optimizer = tf.keras.optimizers.Adam(lr=0.001)
        loss_fn = self.create_confidence_weighted_loss()
        
        self.model.compile(optimizer=optimizer, loss=loss_fn, metrics=['auc'])
        
        # 疑似データでの訓練
        history = self.model.fit(
            pseudo_training_data,
            epochs=50,
            validation_split=0.2,
            callbacks=[
                EarlyStopping(patience=5),
                ReduceLROnPlateau(patience=3),
                ModelCheckpoint('pseudo_user_model.h5')
            ]
        )
        
        return history
    
    def stage2_real_user_adaptation(self, real_user_data, pseudo_model_weights):
        """段階2: 実ユーザーデータでの適応学習"""
        
        print("Stage 2: Real User Adaptation")
        
        # 疑似学習済みモデルをロード
        self.model.load_weights(pseudo_model_weights)
        
        # 実ユーザーデータ用にファインチューニング
        for layer in self.model.layers[:-3]:  # 最後の3層以外は凍結
            layer.trainable = False
        
        # より低い学習率でファインチューニング
        optimizer = tf.keras.optimizers.Adam(lr=0.0001)
        self.model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['auc'])
        
        # 実データでの学習
        history = self.model.fit(
            real_user_data,
            epochs=20,
            validation_split=0.2
        )
        
        return history
    
    def create_confidence_weighted_loss(self):
        """信頼度を考慮した損失関数"""
        def weighted_binary_crossentropy(y_true, y_pred, confidence):
            bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
            weighted_bce = bce * confidence
            return tf.reduce_mean(weighted_bce)
        
        return weighted_binary_crossentropy
```

---

## 🔍 品質保証・検証

### データ品質検証

#### 疑似ユーザーデータの妥当性チェック
```python
class PseudoUserDataValidator:
    def validate_generated_data(self, pseudo_users, user_actions):
        """生成された疑似データの品質検証"""
        
        validation_results = {}
        
        # 1. ユーザー多様性チェック
        validation_results['user_diversity'] = self.check_user_diversity(pseudo_users)
        
        # 2. 行動パターンの現実性チェック
        validation_results['behavior_realism'] = self.check_behavior_realism(user_actions)
        
        # 3. データ分布のバランスチェック  
        validation_results['data_balance'] = self.check_data_balance(user_actions)
        
        # 4. 時系列の一貫性チェック
        validation_results['temporal_consistency'] = self.check_temporal_consistency(user_actions)
        
        return validation_results
    
    def check_user_diversity(self, pseudo_users):
        """ユーザーの多様性検証"""
        user_profiles = [user['profile'] for user in pseudo_users.values()]
        
        return {
            'genre_preference_diversity': self.calculate_diversity(
                [profile.get('preferred_genres', {}) for profile in user_profiles]
            ),
            'rating_pattern_diversity': np.std([
                profile.get('avg_rating', 3.0) for profile in user_profiles
            ]),
            'activity_level_diversity': np.std([
                profile.get('review_count', 0) for profile in user_profiles  
            ])
        }
```

### モデル性能検証

#### 疑似データ学習効果の測定
```python
class PseudoLearningEvaluator:
    def evaluate_pseudo_learning_effectiveness(self, model, test_data):
        """疑似学習の効果を測定"""
        
        # 1. ランダムベースラインとの比較
        random_performance = self.evaluate_random_baseline(test_data)
        model_performance = self.evaluate_model(model, test_data)
        
        improvement = {
            'auc_improvement': model_performance['auc'] - random_performance['auc'],
            'precision_improvement': model_performance['precision@10'] - random_performance['precision@10']
        }
        
        # 2. 人気度ベースラインとの比較  
        popularity_performance = self.evaluate_popularity_baseline(test_data)
        
        # 3. 多様性・新規性の検証
        diversity_metrics = self.calculate_diversity_metrics(model, test_data)
        
        return {
            'performance_improvements': improvement,
            'baseline_comparisons': {
                'vs_random': model_performance['auc'] / random_performance['auc'],
                'vs_popularity': model_performance['auc'] / popularity_performance['auc']
            },
            'diversity_metrics': diversity_metrics
        }
```

---

## 📋 実装計画

### タスク詳細 (フェーズ1拡張)

#### T1.4: レビューデータ大規模収集・変換
```
優先度: 🔴 高
工数見積: 24時間
期間: Week 1-2
```

**サブタスク:**
- [ ] **T1.4.1** - 大規模データ収集システム _(8h)_
  - 10万件レビュー収集自動化
  - 複数サイト並行収集
  - 品質フィルタリング・重複除去
- [ ] **T1.4.2** - 疑似ユーザー生成パイプライン _(8h)_
  - レビュアーID抽出・匿名化
  - ユーザープロファイル生成  
  - 嗜好特徴量抽出
- [ ] **T1.4.3** - 行動データ変換システム _(8h)_
  - 評価値→いいね/スキップ変換
  - 関連動画推定行動生成
  - 負例・時系列データ生成

#### T1.5: ML学習データ準備・検証
```  
優先度: 🔴 高
工数見積: 16時間
期間: Week 2
```

**サブタスク:**
- [ ] **T1.5.1** - 特徴量エンジニアリング _(6h)_
  - ユーザー・アイテム特徴量設計
  - BERT埋め込みベクトル生成
  - カテゴリカル変数エンコーディング
- [ ] **T1.5.2** - データ品質検証システム _(6h)_
  - 疑似データ妥当性チェック
  - 分布バランス検証
  - 現実性スコア算出
- [ ] **T1.5.3** - 訓練用データセット作成 _(4h)_
  - train/validation/test分割
  - NPZ形式での高速保存
  - メタデータ・統計情報生成

### 更新された完了条件

**フェーズ1完了条件 (更新版)**
- [ ] レビューデータ10万件以上収集完了
- [ ] **疑似ユーザー5000人以上生成完了**
- [ ] **ユーザー行動データ50万件以上生成完了**
- [ ] データベーススキーマ全テーブル稼働
- [ ] **疑似データ品質スコア85%以上達成**
- [ ] **ML学習用データセット準備完了**

---

## 🎯 成功指標

### 疑似ユーザー学習の成功指標

#### データ品質指標
- [ ] **ユーザー多様性**: ジャンル嗜好の分散 > 2.0
- [ ] **行動リアリティ**: 実ユーザーパターンとの類似度 > 0.7
- [ ] **データバランス**: Like/Skip比率 40:60 ~ 60:40
- [ ] **時系列一貫性**: セッションパターンの妥当性 > 0.8

#### 学習効果指標  
- [ ] **ベースライン超越**: ランダム推薦に対してAUC +0.15以上
- [ ] **多様性保持**: 推薦結果の多様性指数 > 0.6
- [ ] **冷開始対応**: 新規ユーザーへの推薦精度 > 0.65
- [ ] **実ユーザー適応**: 実データでのファインチューニング効果 +0.05以上

---

**文書管理**  
**作成者**: Claude Code  
**承認者**: -  
**次回レビュー予定**: Week 2完了時