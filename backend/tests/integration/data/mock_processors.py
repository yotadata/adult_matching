"""
データ処理統合テスト用モッククラス
"""
import asyncio
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
import random
import psutil


@dataclass
class PipelineStats:
    """パイプライン統計"""
    processed_items: int = 0
    success_items: int = 0
    failed_items: int = 0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    memory_peak_mb: float = 0.0
    errors: List[str] = field(default_factory=list)
    
    @property
    def duration(self) -> Optional[timedelta]:
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None
    
    @property
    def success_rate(self) -> float:
        if self.processed_items == 0:
            return 0.0
        return self.success_items / self.processed_items


class PipelineManager:
    """モックパイプラインマネージャー"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.stats = PipelineStats()
        self._setup_directories()
    
    def _setup_directories(self):
        """ディレクトリセットアップ"""
        for key, path in self.config.items():
            if key.endswith('_dir') and isinstance(path, Path):
                path.mkdir(parents=True, exist_ok=True)
    
    async def run_full_pipeline(self, input_file: Path) -> PipelineStats:
        """フルパイプライン実行"""
        self.stats = PipelineStats()
        self.stats.start_time = datetime.utcnow()
        
        try:
            # 1. データ読み込み
            raw_data = await self._load_data(input_file)
            
            # 2. データクリーニング
            cleaned_data = await self._clean_data(raw_data)
            
            # 3. データ検証
            validated_data = await self._validate_data(cleaned_data)
            
            # 4. データ変換
            transformed_data = await self._transform_data(validated_data)
            
            # 5. 品質チェック
            quality_score = await self._quality_check(transformed_data)
            
            # 6. 結果保存
            await self._save_results(transformed_data, quality_score)
            
            self.stats.success_items = len(transformed_data)
            self.stats.processed_items = len(raw_data)
            
        except Exception as e:
            self.stats.errors.append(str(e))
            
        finally:
            self.stats.end_time = datetime.utcnow()
            
        return self.stats
    
    async def _load_data(self, input_file: Path) -> List[Dict[str, Any]]:
        """データ読み込み"""
        await asyncio.sleep(0.1)  # I/O待機のシミュレート
        
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return data if isinstance(data, list) else [data]
    
    async def _clean_data(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """データクリーニング"""
        cleaned = []
        
        for item in data:
            # 必須フィールドチェック
            if all(key in item for key in ['content_id', 'user_id', 'rating']):
                # データ正規化
                cleaned_item = {
                    'content_id': str(item['content_id']),
                    'user_id': str(item['user_id']),
                    'rating': float(item.get('rating', 0)),
                    'comment': str(item.get('comment', '')).strip(),
                    'review_date': item.get('review_date', datetime.utcnow().isoformat()),
                    'helpful_count': int(item.get('helpful_count', 0))
                }
                cleaned.append(cleaned_item)
            
            await asyncio.sleep(0.001)  # 処理時間のシミュレート
        
        return cleaned
    
    async def _validate_data(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """データ検証"""
        validated = []
        
        for item in data:
            # バリデーションルール
            if (1.0 <= item['rating'] <= 5.0 and 
                len(item['content_id']) > 0 and
                len(item['user_id']) > 0):
                validated.append(item)
            
            await asyncio.sleep(0.001)
        
        return validated
    
    async def _transform_data(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """データ変換"""
        transformed = []
        
        for item in data:
            # 特徴量生成
            transformed_item = item.copy()
            transformed_item.update({
                'rating_normalized': (item['rating'] - 1.0) / 4.0,  # 0-1正規化
                'comment_length': len(item['comment']),
                'is_helpful': item['helpful_count'] > 0,
                'processed_at': datetime.utcnow().isoformat()
            })
            transformed.append(transformed_item)
            
            await asyncio.sleep(0.002)
        
        return transformed
    
    async def _quality_check(self, data: List[Dict[str, Any]]) -> float:
        """品質チェック"""
        if not data:
            return 0.0
        
        # 品質スコア計算（様々な指標の組み合わせ）
        rating_variance = self._calculate_rating_variance(data)
        completeness_score = self._calculate_completeness(data)
        consistency_score = self._calculate_consistency(data)
        
        # 総合品質スコア
        quality_score = (rating_variance * 0.3 + completeness_score * 0.4 + consistency_score * 0.3)
        
        await asyncio.sleep(0.05)  # 計算時間のシミュレート
        
        return quality_score
    
    def _calculate_rating_variance(self, data: List[Dict[str, Any]]) -> float:
        """評価の分散度合い計算"""
        ratings = [item['rating'] for item in data]
        if len(set(ratings)) > 1:
            return min(1.0, len(set(ratings)) / 5.0)  # 評価のバラエティ
        return 0.5
    
    def _calculate_completeness(self, data: List[Dict[str, Any]]) -> float:
        """データ完全性計算"""
        total_fields = len(data) * 6  # 6つの必須フィールド
        filled_fields = sum(
            sum(1 for value in item.values() if value not in [None, '', 0])
            for item in data
        )
        return filled_fields / total_fields if total_fields > 0 else 0.0
    
    def _calculate_consistency(self, data: List[Dict[str, Any]]) -> float:
        """データ一貫性計算"""
        # 簡単な一貫性チェック（同じユーザーの評価傾向など）
        user_ratings = {}
        for item in data:
            user_id = item['user_id']
            if user_id not in user_ratings:
                user_ratings[user_id] = []
            user_ratings[user_id].append(item['rating'])
        
        # ユーザーごとの評価の一貫性
        consistency_scores = []
        for ratings in user_ratings.values():
            if len(ratings) > 1:
                variance = sum((r - sum(ratings)/len(ratings))**2 for r in ratings) / len(ratings)
                consistency = max(0, 1 - variance / 4.0)  # 分散を0-1スコアに変換
                consistency_scores.append(consistency)
        
        return sum(consistency_scores) / len(consistency_scores) if consistency_scores else 0.8
    
    async def _save_results(self, data: List[Dict[str, Any]], quality_score: float):
        """結果保存"""
        # 処理済みデータ保存
        output_file = self.config['processed_data_dir'] / 'processed_data.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        # 品質レポート保存
        quality_report = {
            'quality_score': quality_score,
            'total_items': len(data),
            'generated_at': datetime.utcnow().isoformat()
        }
        
        quality_file = self.config['processed_data_dir'] / 'quality_report.json'
        with open(quality_file, 'w', encoding='utf-8') as f:
            json.dump(quality_report, f, ensure_ascii=False, indent=2)
        
        await asyncio.sleep(0.05)  # I/O待機のシミュレート


class UnifiedDataProcessor:
    """統合データプロセッサー（パフォーマンステスト用）"""
    
    def __init__(self):
        self.process = psutil.Process()
        self.processing_stats = {
            'items_processed': 0,
            'processing_time': 0.0,
            'memory_peak': 0.0,
            'errors': []
        }
    
    async def process_large_dataset(
        self, 
        dataset: List[Dict[str, Any]], 
        batch_size: int = 1000,
        max_memory_mb: int = 500
    ) -> Dict[str, Any]:
        """大規模データセット処理"""
        start_time = time.time()
        initial_memory = self.process.memory_info().rss / 1024 / 1024
        peak_memory = initial_memory
        
        processed_items = []
        batch_count = 0
        
        try:
            # バッチ処理
            for i in range(0, len(dataset), batch_size):
                batch = dataset[i:i + batch_size]
                batch_count += 1
                
                # バッチ処理
                processed_batch = await self._process_batch(batch)
                processed_items.extend(processed_batch)
                
                # メモリ監視
                current_memory = self.process.memory_info().rss / 1024 / 1024
                peak_memory = max(peak_memory, current_memory)
                
                if current_memory > max_memory_mb:
                    raise MemoryError(f"Memory limit exceeded: {current_memory:.1f}MB > {max_memory_mb}MB")
                
                # 進捗ログ
                if batch_count % 10 == 0:
                    progress = (i + len(batch)) / len(dataset) * 100
                    print(f"Progress: {progress:.1f}% (Memory: {current_memory:.1f}MB)")
                
                # メモリ解放のため短い待機
                await asyncio.sleep(0.001)
        
        except Exception as e:
            self.processing_stats['errors'].append(str(e))
            raise
        
        finally:
            end_time = time.time()
            self.processing_stats.update({
                'items_processed': len(processed_items),
                'processing_time': end_time - start_time,
                'memory_peak': peak_memory,
                'memory_increase': peak_memory - initial_memory
            })
        
        return {
            'processed_data': processed_items,
            'stats': self.processing_stats
        }
    
    async def _process_batch(self, batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """バッチ処理"""
        processed = []
        
        for item in batch:
            # データ処理シミュレート（CPU集約的タスク）
            processed_item = {
                **item,
                'processed_at': datetime.utcnow().isoformat(),
                'hash': hash(str(item)) % 1000000,  # ハッシュ計算
                'features': self._extract_features(item)
            }
            processed.append(processed_item)
        
        # I/O待機のシミュレート
        await asyncio.sleep(0.01)
        
        return processed
    
    def _extract_features(self, item: Dict[str, Any]) -> Dict[str, float]:
        """特徴量抽出（CPU集約的処理のシミュレート）"""
        # 複雑な計算のシミュレート
        rating = float(item.get('rating', 0))
        comment_len = len(str(item.get('comment', '')))
        helpful = int(item.get('helpful_count', 0))
        
        # 特徴量計算
        features = {
            'rating_score': rating / 5.0,
            'comment_score': min(1.0, comment_len / 100.0),
            'helpful_score': min(1.0, helpful / 10.0),
            'composite_score': (rating / 5.0) * 0.5 + min(1.0, helpful / 10.0) * 0.3 + min(1.0, comment_len / 100.0) * 0.2
        }
        
        return features
    
    async def stress_test(
        self, 
        dataset_size: int, 
        concurrent_workers: int = 4,
        target_rps: int = 10
    ) -> Dict[str, Any]:
        """ストレステスト"""
        start_time = time.time()
        
        # 同期データ生成
        test_dataset = [
            {
                'content_id': f'content_{i % 100}',
                'user_id': f'user_{i % 1000}',
                'rating': random.uniform(1.0, 5.0),
                'comment': f'Test comment {i}' * random.randint(1, 5),
                'helpful_count': random.randint(0, 20)
            }
            for i in range(dataset_size)
        ]
        
        # 並行処理
        tasks = []
        batch_size = dataset_size // concurrent_workers
        
        for i in range(concurrent_workers):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size if i < concurrent_workers - 1 else dataset_size
            worker_data = test_dataset[start_idx:end_idx]
            
            task = asyncio.create_task(self._stress_worker(worker_data, i))
            tasks.append(task)
        
        # 全ワーカー完了待機
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        end_time = time.time()
        duration = end_time - start_time
        actual_rps = dataset_size / duration
        
        # 結果集計
        total_processed = sum(r.get('processed', 0) for r in results if isinstance(r, dict))
        total_errors = sum(len(r.get('errors', [])) for r in results if isinstance(r, dict))
        
        return {
            'dataset_size': dataset_size,
            'concurrent_workers': concurrent_workers,
            'target_rps': target_rps,
            'actual_rps': actual_rps,
            'duration': duration,
            'total_processed': total_processed,
            'total_errors': total_errors,
            'performance_ratio': actual_rps / target_rps,
            'worker_results': results
        }
    
    async def _stress_worker(self, data: List[Dict[str, Any]], worker_id: int) -> Dict[str, Any]:
        """ストレステスト用ワーカー"""
        start_time = time.time()
        processed = 0
        errors = []
        
        try:
            for item in data:
                # 処理シミュレート
                features = self._extract_features(item)
                processed += 1
                
                # ランダムエラー注入（5%の確率）
                if random.random() < 0.05:
                    errors.append(f"Worker {worker_id}: Random error at item {processed}")
                
                # CPU負荷シミュレート
                await asyncio.sleep(0.001)
        
        except Exception as e:
            errors.append(f"Worker {worker_id}: {str(e)}")
        
        end_time = time.time()
        
        return {
            'worker_id': worker_id,
            'processed': processed,
            'errors': errors,
            'duration': end_time - start_time,
            'rps': processed / (end_time - start_time) if end_time > start_time else 0
        }


class DataQualityMonitor:
    """データ品質監視"""
    
    def __init__(self):
        self.quality_thresholds = {
            'completeness': 0.95,
            'accuracy': 0.90,
            'consistency': 0.85,
            'timeliness': 0.80
        }
    
    async def comprehensive_quality_check(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """包括的品質チェック"""
        results = {}
        
        # 各品質指標を並行チェック
        tasks = [
            asyncio.create_task(self._check_completeness(data)),
            asyncio.create_task(self._check_accuracy(data)),
            asyncio.create_task(self._check_consistency(data)),
            asyncio.create_task(self._check_timeliness(data))
        ]
        
        completeness, accuracy, consistency, timeliness = await asyncio.gather(*tasks)
        
        results = {
            'completeness': completeness,
            'accuracy': accuracy,
            'consistency': consistency,
            'timeliness': timeliness,
            'overall_score': (completeness + accuracy + consistency + timeliness) / 4,
            'passed_checks': [],
            'failed_checks': []
        }
        
        # 閾値チェック
        for metric, score in results.items():
            if metric in self.quality_thresholds:
                if score >= self.quality_thresholds[metric]:
                    results['passed_checks'].append(metric)
                else:
                    results['failed_checks'].append(metric)
        
        return results
    
    async def _check_completeness(self, data: List[Dict[str, Any]]) -> float:
        """完全性チェック"""
        if not data:
            return 0.0
        
        required_fields = ['content_id', 'user_id', 'rating', 'comment']
        total_checks = len(data) * len(required_fields)
        passed_checks = 0
        
        for item in data:
            for field in required_fields:
                if field in item and item[field] not in [None, '', 0]:
                    passed_checks += 1
        
        await asyncio.sleep(0.01)
        return passed_checks / total_checks
    
    async def _check_accuracy(self, data: List[Dict[str, Any]]) -> float:
        """正確性チェック"""
        if not data:
            return 0.0
        
        valid_items = 0
        
        for item in data:
            # 評価値の範囲チェック
            rating = item.get('rating', 0)
            if 1.0 <= rating <= 5.0:
                # コンテンツIDの形式チェック
                content_id = str(item.get('content_id', ''))
                if len(content_id) > 0 and content_id.replace('_', '').replace('-', '').isalnum():
                    valid_items += 1
        
        await asyncio.sleep(0.01)
        return valid_items / len(data)
    
    async def _check_consistency(self, data: List[Dict[str, Any]]) -> float:
        """一貫性チェック"""
        if not data:
            return 0.0
        
        # ユーザーごとの評価一貫性
        user_ratings = {}
        for item in data:
            user_id = item.get('user_id')
            rating = item.get('rating', 0)
            if user_id:
                if user_id not in user_ratings:
                    user_ratings[user_id] = []
                user_ratings[user_id].append(rating)
        
        consistent_users = 0
        for user_id, ratings in user_ratings.items():
            if len(ratings) > 1:
                # 標準偏差が1以下なら一貫していると判定
                mean_rating = sum(ratings) / len(ratings)
                variance = sum((r - mean_rating) ** 2 for r in ratings) / len(ratings)
                if variance <= 1.0:
                    consistent_users += 1
            else:
                consistent_users += 1  # 単一評価は一貫している
        
        await asyncio.sleep(0.01)
        return consistent_users / len(user_ratings) if user_ratings else 1.0
    
    async def _check_timeliness(self, data: List[Dict[str, Any]]) -> float:
        """適時性チェック"""
        if not data:
            return 0.0
        
        now = datetime.utcnow()
        recent_items = 0
        
        for item in data:
            review_date_str = item.get('review_date', '')
            try:
                # 様々な日付形式に対応
                if review_date_str:
                    if 'T' in review_date_str:
                        review_date = datetime.fromisoformat(review_date_str.replace('Z', '+00:00'))
                    else:
                        review_date = datetime.strptime(review_date_str, '%Y-%m-%d %H:%M:%S')
                    
                    # 1年以内のデータを新しいと判定
                    if (now - review_date).days <= 365:
                        recent_items += 1
                    
            except (ValueError, TypeError):
                # 日付解析エラーは古いデータとして扱う
                pass
        
        await asyncio.sleep(0.01)
        return recent_items / len(data)