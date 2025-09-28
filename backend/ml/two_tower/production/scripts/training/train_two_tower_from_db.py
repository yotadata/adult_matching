#!/usr/bin/env python3
"""
本番データベースからTwo-Towerモデルを学習

本番PostgreSQLデータベースから直接データを取得し、
768次元Two-Towerモデルを訓練する
"""

import os
import sys
import json
import psycopg2
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle

# データベース接続情報
DB_URL = "postgresql://postgres.mfleexehdteobgsyokex:7Jh0iSwSQwXXtc62@aws-1-ap-northeast-1.pooler.supabase.com:5432/postgres"

# 設定
EMBEDDING_DIM = 768
BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = 0.001

def fetch_training_data():
    """本番DBから学習データ取得"""
    print("📊 本番データベースから学習データ取得中...")

    conn = psycopg2.connect(DB_URL)

    # ビデオデータ取得
    videos_query = """
        SELECT
            v.id,
            v.title,
            v.description,
            v.maker,
            v.price,
            v.duration_seconds,
            COALESCE(array_agg(DISTINCT t.name) FILTER (WHERE t.name IS NOT NULL), ARRAY[]::text[]) as tags,
            COALESCE(array_agg(DISTINCT p.name) FILTER (WHERE p.name IS NOT NULL), ARRAY[]::text[]) as performers
        FROM videos v
        LEFT JOIN video_tags vt ON v.id = vt.video_id
        LEFT JOIN tags t ON vt.tag_id = t.id
        LEFT JOIN video_performers vp ON v.id = vp.video_id
        LEFT JOIN performers p ON vp.performer_id = p.id
        GROUP BY v.id, v.title, v.description, v.maker, v.price, v.duration_seconds
        LIMIT 10000
    """

    videos_df = pd.read_sql_query(videos_query, conn)
    print(f"✓ ビデオデータ取得: {len(videos_df)}件")

    # ユーザーインタラクションデータ取得
    interactions_query = """
        SELECT
            uvd.user_id,
            uvd.video_id,
            CASE WHEN uvd.decision_type = 'like' THEN 1 ELSE 0 END as label,
            uvd.created_at
        FROM user_video_decisions uvd
        WHERE uvd.decision_type IN ('like', 'skip')
        ORDER BY uvd.created_at DESC
        LIMIT 50000
    """

    interactions_df = pd.read_sql_query(interactions_query, conn)
    print(f"✓ インタラクションデータ取得: {len(interactions_df)}件")

    # データが少ない場合、疑似インタラクションを生成
    if len(interactions_df) < 1000:
        print(f"⚠️  インタラクションデータが少ないため疑似データ生成中...")

        # 疑似ユーザーID生成
        pseudo_users = [f"pseudo_user_{i}" for i in range(100)]

        # 各疑似ユーザーに対してランダムなインタラクションを生成
        pseudo_interactions = []
        for user_id in pseudo_users:
            # ランダムに10-50本のビデオにインタラクション
            num_interactions = np.random.randint(10, 51)
            sampled_videos = videos_df.sample(n=min(num_interactions, len(videos_df)))

            for _, video in sampled_videos.iterrows():
                # 80%の確率でlike、20%の確率でskip
                decision = 1 if np.random.random() < 0.8 else 0
                pseudo_interactions.append({
                    'user_id': user_id,
                    'video_id': video['id'],
                    'label': decision,
                    'created_at': datetime.now()
                })

        pseudo_df = pd.DataFrame(pseudo_interactions)
        interactions_df = pd.concat([interactions_df, pseudo_df], ignore_index=True)
        print(f"✓ 疑似データ追加後: {len(interactions_df)}件 (実データ: {len(interactions_df) - len(pseudo_df)}件, 疑似: {len(pseudo_df)}件)")

    conn.close()

    return videos_df, interactions_df

def prepare_features(videos_df, interactions_df):
    """特徴量エンジニアリング"""
    print("🔧 特徴量エンジニアリング中...")

    # ユーザーエンコーダー
    user_encoder = LabelEncoder()
    interactions_df['user_idx'] = user_encoder.fit_transform(interactions_df['user_id'])

    # アイテムエンコーダー
    item_encoder = LabelEncoder()
    videos_df['item_idx'] = item_encoder.fit_transform(videos_df['id'])

    # アイテム特徴量作成
    videos_df['title_len'] = videos_df['title'].str.len()
    videos_df['desc_len'] = videos_df['description'].fillna('').str.len()
    videos_df['price_normalized'] = videos_df['price'] / 10000
    videos_df['duration_normalized'] = videos_df['duration_seconds'] / 3600
    videos_df['num_tags'] = videos_df['tags'].apply(len)
    videos_df['num_performers'] = videos_df['performers'].apply(len)

    # メーカーエンコーディング
    maker_encoder = LabelEncoder()
    videos_df['maker_idx'] = maker_encoder.fit_transform(videos_df['maker'].fillna('Unknown'))

    # インタラクションデータとビデオデータをマージ
    merged_df = interactions_df.merge(
        videos_df[['id', 'item_idx', 'price_normalized', 'duration_normalized',
                   'num_tags', 'num_performers', 'maker_idx', 'title_len', 'desc_len']],
        left_on='video_id',
        right_on='id',
        how='inner'
    )

    print(f"✓ マージ後データ: {len(merged_df)}件")

    # ユーザー特徴量作成（集約統計）
    user_stats = merged_df.groupby('user_idx').agg({
        'label': 'sum',
        'price_normalized': 'mean',
        'duration_normalized': 'mean',
        'num_tags': 'mean',
        'num_performers': 'mean'
    }).reset_index()

    user_stats.columns = ['user_idx', 'total_likes', 'avg_price_pref',
                          'avg_duration_pref', 'avg_tags_pref', 'avg_performers_pref']

    # 最終データセット作成
    final_df = merged_df.merge(user_stats, on='user_idx', how='left')

    # 特徴量とラベル分離（float32に変換）
    user_features = final_df[['user_idx', 'total_likes', 'avg_price_pref',
                               'avg_duration_pref', 'avg_tags_pref', 'avg_performers_pref']].values.astype(np.float32)

    item_features = final_df[['item_idx', 'price_normalized', 'duration_normalized',
                               'num_tags', 'num_performers', 'maker_idx',
                               'title_len', 'desc_len']].values.astype(np.float32)

    labels = final_df['label'].values.astype(np.float32)

    print(f"✓ ユーザー特徴量: {user_features.shape}")
    print(f"✓ アイテム特徴量: {item_features.shape}")
    print(f"✓ ラベル: {labels.shape}")

    encoders = {
        'user_encoder': user_encoder,
        'item_encoder': item_encoder,
        'maker_encoder': maker_encoder,
        'num_users': len(user_encoder.classes_),
        'num_items': len(item_encoder.classes_),
        'num_makers': len(maker_encoder.classes_)
    }

    return user_features, item_features, labels, encoders

def build_two_tower_model(num_users, num_items, embedding_dim=768):
    """768次元Two-Towerモデル構築"""
    print(f"🏗️  {embedding_dim}次元Two-Towerモデル構築中...")

    # ユーザータワー
    user_input = layers.Input(shape=(6,), name='user_features')
    user_dense1 = layers.Dense(256, activation='relu')(user_input)
    user_bn1 = layers.BatchNormalization()(user_dense1)
    user_dropout1 = layers.Dropout(0.3)(user_bn1)
    user_dense2 = layers.Dense(512, activation='relu')(user_dropout1)
    user_bn2 = layers.BatchNormalization()(user_dense2)
    user_dropout2 = layers.Dropout(0.3)(user_bn2)
    user_embedding = layers.Dense(embedding_dim, activation='relu', name='user_embedding')(user_dropout2)

    # アイテムタワー
    item_input = layers.Input(shape=(8,), name='item_features')
    item_dense1 = layers.Dense(256, activation='relu')(item_input)
    item_bn1 = layers.BatchNormalization()(item_dense1)
    item_dropout1 = layers.Dropout(0.3)(item_bn1)
    item_dense2 = layers.Dense(512, activation='relu')(item_dropout1)
    item_bn2 = layers.BatchNormalization()(item_dense2)
    item_dropout2 = layers.Dropout(0.3)(item_bn2)
    item_embedding = layers.Dense(embedding_dim, activation='relu', name='item_embedding')(item_dropout2)

    # ドット積
    dot_product = layers.Dot(axes=1, normalize=True)([user_embedding, item_embedding])

    # 出力層
    output = layers.Dense(1, activation='sigmoid')(dot_product)

    # モデル作成
    model = models.Model(inputs=[user_input, item_input], outputs=output)

    # ユーザータワーとアイテムタワーを個別モデルとして抽出
    user_tower = models.Model(inputs=user_input, outputs=user_embedding, name='user_tower')
    item_tower = models.Model(inputs=item_input, outputs=item_embedding, name='item_tower')

    print(f"✓ Full Model: {model.count_params():,} パラメータ")
    print(f"✓ User Tower: {user_tower.count_params():,} パラメータ")
    print(f"✓ Item Tower: {item_tower.count_params():,} パラメータ")

    return model, user_tower, item_tower

def train_model(model, user_features, item_features, labels):
    """モデル訓練"""
    print("🚀 モデル訓練開始...")

    # データ分割
    X_user_train, X_user_val, X_item_train, X_item_val, y_train, y_val = train_test_split(
        user_features, item_features, labels, test_size=0.2, random_state=42
    )

    # コンパイル
    model.compile(
        optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )

    # コールバック
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=2)
    ]

    # 訓練
    history = model.fit(
        [X_user_train, X_item_train],
        y_train,
        validation_data=([X_user_val, X_item_val], y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )

    # 評価
    val_loss, val_acc, val_auc = model.evaluate([X_user_val, X_item_val], y_val, verbose=0)
    print(f"\n✓ 検証損失: {val_loss:.4f}")
    print(f"✓ 検証精度: {val_acc:.4f}")
    print(f"✓ 検証AUC: {val_auc:.4f}")

    return history

def save_models(model, user_tower, item_tower, encoders):
    """モデル保存"""
    print("💾 モデル保存中...")

    # 保存ディレクトリ作成
    save_dir = Path("/home/devel/dev/adult_matching/backend/ml/models/two_tower_768_production")
    save_dir.mkdir(parents=True, exist_ok=True)

    # モデル保存
    model.save(save_dir / "full_model_768.keras")
    user_tower.save(save_dir / "user_tower_768.keras")
    item_tower.save(save_dir / "item_tower_768.keras")

    # エンコーダー保存
    with open(save_dir / "encoders_768.pkl", "wb") as f:
        pickle.dump(encoders, f)

    # メタデータ保存
    metadata = {
        "model_type": "two_tower_768",
        "embedding_dim": EMBEDDING_DIM,
        "trained_at": datetime.now().isoformat(),
        "num_users": encoders['num_users'],
        "num_items": encoders['num_items'],
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE
    }

    with open(save_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"✓ モデル保存完了: {save_dir}")

def main():
    print("=" * 60)
    print("🤖 768次元Two-Towerモデル訓練（本番DB使用）")
    print("=" * 60)

    # データ取得
    videos_df, interactions_df = fetch_training_data()

    # 特徴量準備
    user_features, item_features, labels, encoders = prepare_features(videos_df, interactions_df)

    # モデル構築
    model, user_tower, item_tower = build_two_tower_model(
        encoders['num_users'],
        encoders['num_items'],
        EMBEDDING_DIM
    )

    # 訓練
    history = train_model(model, user_features, item_features, labels)

    # 保存
    save_models(model, user_tower, item_tower, encoders)

    print("\n🎉 訓練完了！")
    print("=" * 60)

if __name__ == "__main__":
    main()