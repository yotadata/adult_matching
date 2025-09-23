"""
Pydantic Data Models

Pydanticベースのデータモデル定義 - 型安全性とバリデーション
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, validator
import numpy as np


class DataSourceType(str, Enum):
    """データソース種別"""
    DMM = "dmm"
    FANZA = "fanza"
    MANUAL = "manual"
    SCRAPED = "scraped"
    API = "api"


class VideoStatus(str, Enum):
    """動画ステータス"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    DELETED = "deleted"
    PROCESSING = "processing"


class UserStatus(str, Enum):
    """ユーザーステータス"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"
    DELETED = "deleted"


class ValidationStatus(str, Enum):
    """検証ステータス"""
    PENDING = "pending"
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"


class VideoDataSchema(BaseModel):
    """動画データスキーマ"""
    
    # 基本情報
    external_id: str = Field(..., min_length=1, max_length=100, description="外部ID")
    title: str = Field(..., min_length=1, max_length=500, description="タイトル")
    source: DataSourceType = Field(..., description="データソース")
    status: VideoStatus = Field(default=VideoStatus.ACTIVE, description="ステータス")
    
    # メタデータ
    description: Optional[str] = Field(None, max_length=5000, description="説明文")
    duration: Optional[int] = Field(None, ge=0, le=14400, description="再生時間（秒）")
    price: Optional[int] = Field(None, ge=0, description="価格")
    rating: Optional[float] = Field(None, ge=0.0, le=5.0, description="評価")
    
    # メディア情報
    thumbnail_url: Optional[str] = Field(None, description="サムネイルURL")
    video_url: Optional[str] = Field(None, description="動画URL")
    
    # カテゴリ・タグ
    genre: Optional[str] = Field(None, max_length=100, description="ジャンル")
    tags: Optional[List[str]] = Field(default_factory=list, description="タグリスト")
    maker: Optional[str] = Field(None, max_length=200, description="制作会社")
    performers: Optional[List[str]] = Field(default_factory=list, description="出演者リスト")
    
    # 統計情報
    view_count: Optional[int] = Field(None, ge=0, description="視聴回数")
    like_count: Optional[int] = Field(None, ge=0, description="いいね数")
    
    # タイムスタンプ
    created_at: datetime = Field(default_factory=datetime.now, description="作成日時")
    updated_at: datetime = Field(default_factory=datetime.now, description="更新日時")
    
    # 追加メタデータ
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="追加メタデータ")
    
    @validator('external_id')
    def validate_external_id(cls, v):
        if not v or v.strip() == '':
            raise ValueError('外部IDは空にできません')
        return v.strip()
    
    @validator('title')
    def validate_title(cls, v):
        if not v or v.strip() == '':
            raise ValueError('タイトルは空にできません')
        return v.strip()
    
    @validator('tags', pre=True)
    def validate_tags(cls, v):
        if v is None:
            return []
        if isinstance(v, str):
            return [tag.strip() for tag in v.split(',') if tag.strip()]
        return v
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class UserDataSchema(BaseModel):
    """ユーザーデータスキーマ"""
    
    # 基本情報
    user_id: str = Field(..., min_length=1, description="ユーザーID")
    email: Optional[str] = Field(None, description="メールアドレス")
    status: UserStatus = Field(default=UserStatus.ACTIVE, description="ステータス")
    
    # プロフィール情報
    username: Optional[str] = Field(None, max_length=50, description="ユーザー名")
    age: Optional[int] = Field(None, ge=18, le=100, description="年齢")
    gender: Optional[str] = Field(None, description="性別")
    
    # 設定
    preferences: Optional[Dict[str, Any]] = Field(default_factory=dict, description="設定")
    privacy_settings: Optional[Dict[str, bool]] = Field(default_factory=dict, description="プライバシー設定")
    
    # 統計情報
    total_likes: Optional[int] = Field(None, ge=0, description="総いいね数")
    total_views: Optional[int] = Field(None, ge=0, description="総視聴数")
    last_active: Optional[datetime] = Field(None, description="最終アクティブ日時")
    
    # タイムスタンプ
    created_at: datetime = Field(default_factory=datetime.now, description="作成日時")
    updated_at: datetime = Field(default_factory=datetime.now, description="更新日時")
    
    @validator('email')
    def validate_email(cls, v):
        if v and '@' not in v:
            raise ValueError('有効なメールアドレスを入力してください')
        return v
    
    @validator('user_id')
    def validate_user_id(cls, v):
        if not v or v.strip() == '':
            raise ValueError('ユーザーIDは空にできません')
        return v.strip()
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None
        }


class ReviewDataSchema(BaseModel):
    """レビューデータスキーマ"""
    
    # 基本情報
    review_id: str = Field(..., description="レビューID")
    external_video_id: str = Field(..., description="動画外部ID")
    content_id: Optional[str] = Field(None, description="コンテンツID")
    
    # レビュー内容
    rating: Optional[float] = Field(None, ge=0.0, le=5.0, description="評価")
    review_text: Optional[str] = Field(None, max_length=10000, description="レビューテキスト")
    
    # 統計情報
    helpful_count: Optional[int] = Field(None, ge=0, description="参考になった数")
    total_votes: Optional[int] = Field(None, ge=0, description="総投票数")
    
    # メタデータ
    reviewer_info: Optional[Dict[str, Any]] = Field(default_factory=dict, description="レビュアー情報")
    review_metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="レビューメタデータ")
    
    # タイムスタンプ
    review_date: Optional[datetime] = Field(None, description="レビュー日時")
    created_at: datetime = Field(default_factory=datetime.now, description="作成日時")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None
        }


class EmbeddingDataSchema(BaseModel):
    """埋め込みデータスキーマ"""
    
    # 基本情報
    embedding_id: str = Field(..., description="埋め込みID")
    entity_id: str = Field(..., description="エンティティID")
    entity_type: str = Field(..., description="エンティティタイプ（user/video/content）")
    
    # 埋め込み情報
    embedding_vector: List[float] = Field(..., description="埋め込みベクトル")
    dimension: int = Field(..., ge=1, description="次元数")
    model_name: str = Field(..., description="モデル名")
    model_version: str = Field(..., description="モデルバージョン")
    
    # メタデータ
    generation_method: str = Field(..., description="生成方法")
    confidence_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="信頼度スコア")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="メタデータ")
    
    # タイムスタンプ
    created_at: datetime = Field(default_factory=datetime.now, description="作成日時")
    updated_at: datetime = Field(default_factory=datetime.now, description="更新日時")
    
    @validator('embedding_vector')
    def validate_embedding_vector(cls, v):
        if not v:
            raise ValueError('埋め込みベクトルは空にできません')
        if len(v) == 0:
            raise ValueError('埋め込みベクトルは空のリストにできません')
        return v
    
    @validator('dimension')
    def validate_dimension(cls, v, values):
        if 'embedding_vector' in values and len(values['embedding_vector']) != v:
            raise ValueError('次元数と埋め込みベクトルの長さが一致しません')
        return v
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ValidationResultSchema(BaseModel):
    """検証結果スキーマ"""
    
    # 基本情報
    validation_id: str = Field(..., description="検証ID")
    data_source: str = Field(..., description="データソース")
    validation_type: str = Field(..., description="検証タイプ")
    
    # 検証結果
    status: ValidationStatus = Field(..., description="検証ステータス")
    overall_score: float = Field(..., ge=0.0, le=1.0, description="総合スコア")
    passed: bool = Field(..., description="検証合格")
    
    # 詳細結果
    total_records: int = Field(..., ge=0, description="総レコード数")
    processed_records: int = Field(..., ge=0, description="処理済みレコード数")
    error_count: int = Field(..., ge=0, description="エラー数")
    warning_count: int = Field(..., ge=0, description="警告数")
    
    # エラー・警告詳細
    errors: List[str] = Field(default_factory=list, description="エラーリスト")
    warnings: List[str] = Field(default_factory=list, description="警告リスト")
    
    # 統計情報
    field_statistics: Optional[Dict[str, Any]] = Field(default_factory=dict, description="フィールド統計")
    quality_metrics: Optional[Dict[str, float]] = Field(default_factory=dict, description="品質メトリクス")
    
    # 推奨事項
    recommendations: List[str] = Field(default_factory=list, description="改善推奨事項")
    
    # タイムスタンプ
    validation_start: datetime = Field(..., description="検証開始時刻")
    validation_end: Optional[datetime] = Field(None, description="検証終了時刻")
    created_at: datetime = Field(default_factory=datetime.now, description="作成日時")
    
    @property
    def validation_duration_seconds(self) -> Optional[float]:
        """検証実行時間"""
        if self.validation_end:
            return (self.validation_end - self.validation_start).total_seconds()
        return None
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class BatchProcessingSchema(BaseModel):
    """バッチ処理スキーマ"""
    
    # 基本情報
    batch_id: str = Field(..., description="バッチID")
    process_type: str = Field(..., description="処理タイプ")
    status: str = Field(..., description="処理ステータス")
    
    # 処理設定
    config: Dict[str, Any] = Field(..., description="処理設定")
    input_sources: List[str] = Field(..., description="入力ソース")
    output_targets: List[str] = Field(..., description="出力ターゲット")
    
    # 進捗情報
    total_items: Optional[int] = Field(None, ge=0, description="総アイテム数")
    processed_items: Optional[int] = Field(None, ge=0, description="処理済みアイテム数")
    failed_items: Optional[int] = Field(None, ge=0, description="失敗アイテム数")
    
    # タイムスタンプ
    started_at: datetime = Field(..., description="開始時刻")
    completed_at: Optional[datetime] = Field(None, description="完了時刻")
    
    @property
    def completion_percentage(self) -> Optional[float]:
        """完了パーセンテージ"""
        if self.total_items and self.total_items > 0:
            return (self.processed_items or 0) / self.total_items * 100
        return None
    
    @property
    def success_rate(self) -> Optional[float]:
        """成功率"""
        processed = self.processed_items or 0
        failed = self.failed_items or 0
        total_attempted = processed + failed
        
        if total_attempted > 0:
            return processed / total_attempted * 100
        return None
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class APIResponseSchema(BaseModel):
    """API応答スキーマ"""
    
    success: bool = Field(..., description="成功フラグ")
    message: str = Field(..., description="メッセージ")
    data: Optional[Any] = Field(None, description="データ")
    errors: Optional[List[str]] = Field(None, description="エラーリスト")
    metadata: Optional[Dict[str, Any]] = Field(None, description="メタデータ")
    timestamp: datetime = Field(default_factory=datetime.now, description="タイムスタンプ")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }