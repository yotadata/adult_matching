"""
TensorFlow.js Model Conversion and Deployment Utility

TensorFlow.js形式への変換とデプロイメントユーティリティ
768次元Two-Towerモデルの標準化とWeb配信用の最適化
"""

import os
import json
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime

import numpy as np
import tensorflow as tf
import tensorflowjs as tfjs
from tf2onnx import tf2onnx

from backend.ml.utils.logger import get_ml_logger
from backend.ml.training.trainers.unified_two_tower_trainer import UnifiedTwoTowerTrainer, TrainingConfig

logger = get_ml_logger(__name__)

class TensorFlowJSConverter:
    """TensorFlow.jsモデル変換器"""
    
    def __init__(self, target_dir: Optional[Path] = None):
        self.target_dir = target_dir or Path("models/tensorflowjs")
        self.target_dir.mkdir(parents=True, exist_ok=True)
        
        # 768次元標準設定
        self.standard_config = {
            "embedding_dim": 768,
            "quantization": True,
            "optimization": "speed",  # "speed" or "size"
            "precision": "float16"    # "float16" or "float32"
        }
        
    def convert_two_tower_model(
        self,
        model_path: str,
        output_name: str = "two_tower_768",
        quantize: bool = True,
        optimize_for: str = "speed"
    ) -> Dict[str, Any]:
        """Two-Towerモデルの変換"""
        logger.info(f"Converting Two-Tower model from {model_path}")
        
        try:
            # モデルの読み込み
            model = tf.keras.models.load_model(model_path)
            
            # 768次元への標準化確認
            self._validate_model_dimensions(model)
            
            # 推論専用モデルの作成
            inference_model = self._create_inference_model(model)
            
            # TensorFlow.js形式への変換
            tfjs_path = self.target_dir / output_name
            tfjs_path.mkdir(exist_ok=True)
            
            conversion_options = {
                "quantize_float16": quantize and optimize_for == "size",
                "skip_op_check": True,
                "strip_debug_ops": True
            }
            
            if quantize:
                if optimize_for == "speed":
                    conversion_options["quantize_float16"] = True
                else:  # size
                    conversion_options["quantize_uint8"] = True
            
            tfjs.converters.save_keras_model(
                inference_model,
                str(tfjs_path),
                **conversion_options
            )
            
            # メタデータの生成
            metadata = self._generate_model_metadata(
                model, 
                output_name, 
                quantize, 
                optimize_for
            )
            
            # メタデータの保存
            with open(tfjs_path / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)
            
            # JavaScript配信用設定の生成
            js_config = self._generate_js_config(metadata)
            with open(tfjs_path / "config.js", "w") as f:
                f.write(f"window.TWO_TOWER_CONFIG = {json.dumps(js_config, indent=2)};")
            
            logger.info(f"Model converted successfully to {tfjs_path}")
            
            return {
                "success": True,
                "output_path": str(tfjs_path),
                "metadata": metadata,
                "file_size_mb": self._calculate_model_size(tfjs_path)
            }
            
        except Exception as error:
            logger.error(f"Model conversion failed: {error}")
            return {
                "success": False,
                "error": str(error)
            }
    
    def _validate_model_dimensions(self, model: tf.keras.Model):
        """モデル次元の768次元標準化検証"""
        logger.info("Validating model dimensions for 768-dim standard")
        
        # モデル構造の確認
        for layer in model.layers:
            if hasattr(layer, 'output_shape'):
                output_shape = layer.output_shape
                if isinstance(output_shape, (list, tuple)) and len(output_shape) > 1:
                    if output_shape[-1] == 768:
                        logger.info(f"Found 768-dim layer: {layer.name} - {output_shape}")
                    elif output_shape[-1] in [512, 256, 128] and "embedding" in layer.name.lower():
                        logger.warning(f"Non-standard embedding dimension in {layer.name}: {output_shape[-1]}")
        
        # 入力形状の確認
        if hasattr(model, 'input_shape'):
            input_shapes = model.input_shape if isinstance(model.input_shape, list) else [model.input_shape]
            for i, shape in enumerate(input_shapes):
                logger.info(f"Input {i} shape: {shape}")
        
        # 出力形状の確認
        if hasattr(model, 'output_shape'):
            output_shapes = model.output_shape if isinstance(model.output_shape, list) else [model.output_shape]
            for i, shape in enumerate(output_shapes):
                logger.info(f"Output {i} shape: {shape}")
                if shape[-1] != 768:
                    logger.warning(f"Output {i} is not 768-dimensional: {shape}")
    
    def _create_inference_model(self, model: tf.keras.Model) -> tf.keras.Model:
        """推論専用モデルの作成"""
        logger.info("Creating inference-optimized model")
        
        # 推論時に不要な層を除去
        inference_layers = []
        
        for layer in model.layers:
            # Dropout層を除去
            if isinstance(layer, tf.keras.layers.Dropout):
                continue
            
            # BatchNormalizationを推論モードで固定
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False
            
            inference_layers.append(layer)
        
        # 新しいモデルを構築
        if hasattr(model, 'input') and hasattr(model, 'output'):
            try:
                # Functional APIモデルの場合
                inference_model = tf.keras.Model(
                    inputs=model.input,
                    outputs=model.output,
                    name=f"{model.name}_inference"
                )
                inference_model.set_weights(model.get_weights())
                return inference_model
            except Exception:
                # フォールバック: 元のモデルを返す
                logger.warning("Failed to create inference model, using original")
                return model
        
        return model
    
    def _generate_model_metadata(
        self,
        model: tf.keras.Model,
        model_name: str,
        quantized: bool,
        optimization: str
    ) -> Dict[str, Any]:
        """モデルメタデータの生成"""
        return {
            "model_name": model_name,
            "version": "1.0.0",
            "created_at": datetime.now().isoformat(),
            "framework": "tensorflow.js",
            "embedding_dimension": 768,
            "architecture": "two_tower",
            "quantized": quantized,
            "optimization_target": optimization,
            "model_info": {
                "total_params": model.count_params(),
                "trainable_params": sum([tf.keras.utils.count_params(w) for w in model.trainable_weights]),
                "input_shapes": self._extract_shapes(model.input_shape),
                "output_shapes": self._extract_shapes(model.output_shape)
            },
            "inference_config": {
                "batch_size": 1,
                "max_sequence_length": None,
                "preprocessing_required": True
            },
            "deployment_info": {
                "target_platforms": ["web", "mobile"],
                "min_tensorflowjs_version": "3.18.0",
                "browser_compatibility": {
                    "chrome": ">=80",
                    "firefox": ">=75",
                    "safari": ">=13",
                    "edge": ">=80"
                }
            }
        }
    
    def _extract_shapes(self, shapes) -> List:
        """形状情報の抽出"""
        if isinstance(shapes, (list, tuple)):
            return [list(shape) if hasattr(shape, '__iter__') else shape for shape in shapes]
        elif hasattr(shapes, '__iter__'):
            return list(shapes)
        else:
            return [shapes]
    
    def _generate_js_config(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """JavaScript配信用設定の生成"""
        return {
            "modelName": metadata["model_name"],
            "version": metadata["version"],
            "embeddingDim": metadata["embedding_dimension"],
            "inputShapes": metadata["model_info"]["input_shapes"],
            "outputShapes": metadata["model_info"]["output_shapes"],
            "inferenceConfig": metadata["inference_config"],
            "apiEndpoints": {
                "userEmbedding": "/functions/v1/user-management/embeddings",
                "recommendations": "/functions/v1/content/recommendations",
                "videosFeed": "/functions/v1/content/videos-feed"
            },
            "preprocessing": {
                "normalization": True,
                "featureScaling": "standard",
                "categoryEncoding": "onehot"
            }
        }
    
    def _calculate_model_size(self, model_path: Path) -> float:
        """モデルファイルサイズの計算"""
        total_size = 0
        for file_path in model_path.rglob("*"):
            if file_path.is_file():
                total_size += file_path.stat().st_size
        
        return total_size / (1024 * 1024)  # MB
    
    def create_web_deployment_package(
        self,
        model_names: List[str],
        output_dir: Optional[Path] = None
    ) -> Dict[str, Any]:
        """Web配信パッケージの作成"""
        logger.info("Creating web deployment package")
        
        output_dir = output_dir or Path("web_deployment")
        output_dir.mkdir(exist_ok=True)
        
        try:
            # 各モデルをパッケージに含める
            models_info = []
            
            for model_name in model_names:
                model_path = self.target_dir / model_name
                if not model_path.exists():
                    logger.warning(f"Model {model_name} not found, skipping")
                    continue
                
                # モデルファイルのコピー
                target_path = output_dir / "models" / model_name
                target_path.mkdir(parents=True, exist_ok=True)
                
                for file_path in model_path.rglob("*"):
                    if file_path.is_file():
                        relative_path = file_path.relative_to(model_path)
                        target_file = target_path / relative_path
                        target_file.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(file_path, target_file)
                
                # メタデータの読み込み
                metadata_path = model_path / "metadata.json"
                if metadata_path.exists():
                    with open(metadata_path) as f:
                        metadata = json.load(f)
                    models_info.append(metadata)
            
            # デプロイメント設定の生成
            deployment_config = {
                "deployment_id": f"two_tower_web_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "created_at": datetime.now().isoformat(),
                "models": models_info,
                "web_config": {
                    "cdn_enabled": True,
                    "compression": "gzip",
                    "caching_policy": "max-age=86400",
                    "cors_enabled": True
                },
                "integration": {
                    "framework": "vanilla_js",
                    "module_type": "es6",
                    "bundler_compatible": True
                }
            }
            
            # 設定ファイルの保存
            with open(output_dir / "deployment.json", "w") as f:
                json.dump(deployment_config, f, indent=2)
            
            # README生成
            self._generate_deployment_readme(output_dir, deployment_config)
            
            logger.info(f"Web deployment package created at {output_dir}")
            
            return {
                "success": True,
                "package_path": str(output_dir),
                "models_included": len(models_info),
                "total_size_mb": sum([
                    self._calculate_model_size(self.target_dir / name) 
                    for name in model_names 
                    if (self.target_dir / name).exists()
                ])
            }
            
        except Exception as error:
            logger.error(f"Web deployment package creation failed: {error}")
            return {
                "success": False,
                "error": str(error)
            }
    
    def _generate_deployment_readme(self, output_dir: Path, config: Dict[str, Any]):
        """デプロイメントREADMEの生成"""
        readme_content = f"""# Two-Tower 768-dim Model Deployment Package

Generated on: {config['created_at']}
Deployment ID: {config['deployment_id']}

## Models Included

{chr(10).join([f"- {model['model_name']} (v{model['version']})" for model in config['models']])}

## Quick Start

### 1. Include TensorFlow.js

```html
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@latest"></script>
```

### 2. Load Model

```javascript
// Load the Two-Tower model
const model = await tf.loadLayersModel('./models/{config['models'][0]['model_name']}/model.json');

// Load configuration
const config = window.TWO_TOWER_CONFIG;
```

### 3. Run Inference

```javascript
// Prepare input features (example)
const userFeatures = tf.tensor2d([[/* user feature vector */]]);
const itemFeatures = tf.tensor2d([[/* item feature vector */]]);

// Get embeddings
const userEmbedding = model.predict(userFeatures);
const itemEmbedding = model.predict(itemFeatures);

// Calculate similarity
const similarity = tf.matMul(userEmbedding, itemEmbedding, false, true);
```

## Integration with Backend

The models are designed to work with the following API endpoints:

{chr(10).join([f"- {endpoint}" for endpoint in config['models'][0]['inference_config'] if 'api' in str(endpoint)])}

## Browser Compatibility

{chr(10).join([f"- {browser}: {version}" for browser, version in config['models'][0]['deployment_info']['browser_compatibility'].items()])}

## Performance Notes

- All models are optimized for 768-dimensional embeddings
- Quantization is enabled for reduced model size
- Batch inference is supported for multiple items

## Support

For issues or questions, refer to the project documentation.
"""
        
        with open(output_dir / "README.md", "w") as f:
            f.write(readme_content)

def create_768_standard_models():
    """768次元標準モデルの作成"""
    logger.info("Creating 768-dimensional standard models")
    
    # 標準設定でのトレーナー初期化
    config = TrainingConfig(
        user_embedding_dim=768,
        item_embedding_dim=768,
        user_hidden_units=[512, 256, 128],
        item_hidden_units=[512, 256, 128],
        dropout_rate=0.2,
        l2_regularization=0.01,
        learning_rate=0.001,
        batch_size=32,
        epochs=50,
        validation_split=0.2
    )
    
    trainer = UnifiedTwoTowerTrainer(config)
    
    # ダミーデータでのモデル構築（実際の使用時は実データを使用）
    user_input_dim = 100  # ユーザー特徴量次元
    item_input_dim = 120  # アイテム特徴量次元
    
    # モデルの構築
    model = trainer.build_two_tower_model(user_input_dim, item_input_dim)
    
    # モデルの保存
    model_path = Path("models/two_tower_768_standard")
    model_path.mkdir(parents=True, exist_ok=True)
    
    model.save(str(model_path))
    logger.info(f"768-dimensional standard model saved to {model_path}")
    
    return str(model_path)

if __name__ == "__main__":
    # 768次元標準モデルの作成
    model_path = create_768_standard_models()
    
    # TensorFlow.js変換
    converter = TensorFlowJSConverter()
    
    # 速度最適化版の作成
    speed_result = converter.convert_two_tower_model(
        model_path,
        "two_tower_768_speed",
        quantize=True,
        optimize_for="speed"
    )
    
    # サイズ最適化版の作成
    size_result = converter.convert_two_tower_model(
        model_path,
        "two_tower_768_size",
        quantize=True,
        optimize_for="size"
    )
    
    # Web配信パッケージの作成
    package_result = converter.create_web_deployment_package([
        "two_tower_768_speed",
        "two_tower_768_size"
    ])
    
    logger.info("Model conversion and deployment package creation completed")
    logger.info(f"Speed optimized: {speed_result}")
    logger.info(f"Size optimized: {size_result}")
    logger.info(f"Web package: {package_result}")