#!/usr/bin/env python3
"""
768-Dimensional Model Standardization Script

既存モデルの768次元標準への変換・検証・統合スクリプト
TensorFlow.js変換とWeb配信パッケージ自動生成
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

# 親ディレクトリをPythonパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from backend.ml import (
    STANDARD_MODEL_CONFIG,
    DEFAULT_EMBEDDING_DIM,
    TENSORFLOWJS_CONFIG,
    create_standard_trainer,
    create_tensorflowjs_converter
)
from backend.ml.utils.logger import get_ml_logger
from backend.ml.training.trainers.unified_two_tower_trainer import TrainingConfig
from backend.ml.preprocessing import FeatureProcessor, FeatureConfig
from backend.ml.deployment.tensorflowjs_converter import create_768_standard_models

logger = get_ml_logger(__name__)

def validate_existing_models(models_dir: Path) -> Dict[str, Dict]:
    """既存モデルの768次元適合性検証"""
    logger.info(f"Validating existing models in {models_dir}")
    
    validation_results = {}
    
    if not models_dir.exists():
        logger.warning(f"Models directory {models_dir} not found")
        return validation_results
    
    # 各モデルディレクトリをチェック
    for model_path in models_dir.iterdir():
        if not model_path.is_dir():
            continue
            
        model_name = model_path.name
        logger.info(f"Validating model: {model_name}")
        
        try:
            # モデルファイルの存在確認
            model_files = list(model_path.glob("*.h5")) + list(model_path.glob("*.keras"))
            
            if not model_files:
                # SavedModel形式の確認
                saved_model_path = model_path / "saved_model.pb"
                if saved_model_path.exists():
                    model_files = [saved_model_path]
            
            if not model_files:
                validation_results[model_name] = {
                    "status": "error",
                    "message": "No model files found"
                }
                continue
            
            # モデル読み込み試行
            try:
                import tensorflow as tf
                
                model_file = model_files[0]
                if model_file.suffix in ['.h5', '.keras']:
                    model = tf.keras.models.load_model(str(model_file))
                else:
                    model = tf.saved_model.load(str(model_path))
                    model = model.signatures['serving_default']
                
                # 入出力形状の確認
                input_shapes = []
                output_shapes = []
                
                if hasattr(model, 'input_shape'):
                    input_shapes = [model.input_shape] if not isinstance(model.input_shape, list) else model.input_shape
                
                if hasattr(model, 'output_shape'):
                    output_shapes = [model.output_shape] if not isinstance(model.output_shape, list) else model.output_shape
                
                # 768次元の検証
                is_768_compliant = False
                embedding_layers = []
                
                for layer in model.layers if hasattr(model, 'layers') else []:
                    if hasattr(layer, 'output_shape') and layer.output_shape:
                        output_shape = layer.output_shape
                        if isinstance(output_shape, (tuple, list)) and len(output_shape) > 1:
                            if output_shape[-1] == 768:
                                is_768_compliant = True
                                embedding_layers.append({
                                    "name": layer.name,
                                    "shape": output_shape,
                                    "type": type(layer).__name__
                                })
                
                validation_results[model_name] = {
                    "status": "success",
                    "is_768_compliant": is_768_compliant,
                    "input_shapes": [list(shape) if hasattr(shape, '__iter__') else shape for shape in input_shapes],
                    "output_shapes": [list(shape) if hasattr(shape, '__iter__') else shape for shape in output_shapes],
                    "embedding_layers": embedding_layers,
                    "total_params": model.count_params() if hasattr(model, 'count_params') else "unknown",
                    "model_file": str(model_file)
                }
                
            except Exception as load_error:
                validation_results[model_name] = {
                    "status": "load_error",
                    "message": str(load_error)
                }
                
        except Exception as error:
            validation_results[model_name] = {
                "status": "error", 
                "message": str(error)
            }
    
    return validation_results

def convert_models_to_768(validation_results: Dict[str, Dict], models_dir: Path) -> Dict[str, Dict]:
    """非768次元モデルの768次元への変換"""
    logger.info("Converting non-768 models to 768-dimensional standard")
    
    conversion_results = {}
    
    for model_name, validation in validation_results.items():
        if validation["status"] != "success":
            logger.info(f"Skipping {model_name} due to validation failure")
            continue
            
        if validation["is_768_compliant"]:
            logger.info(f"Model {model_name} already 768-compliant")
            conversion_results[model_name] = {
                "status": "already_compliant",
                "action": "none"
            }
            continue
        
        logger.info(f"Converting {model_name} to 768-dimensional standard")
        
        try:
            # 新しい768次元モデルの作成
            config = TrainingConfig(**STANDARD_MODEL_CONFIG)
            trainer = create_standard_trainer(config=config.__dict__)
            
            # 元のモデルの特徴量次元を推定
            input_shapes = validation["input_shapes"]
            if input_shapes and len(input_shapes) >= 2:
                user_dim = input_shapes[0][-1] if input_shapes[0] else 100
                item_dim = input_shapes[1][-1] if len(input_shapes) > 1 and input_shapes[1] else 120
            else:
                user_dim, item_dim = 100, 120  # デフォルト値
            
            # 新しい768次元モデルの構築
            new_model = trainer.build_two_tower_model(user_dim, item_dim)
            
            # モデルの保存
            new_model_path = models_dir / f"{model_name}_768_standardized"
            new_model_path.mkdir(exist_ok=True)
            
            new_model.save(str(new_model_path / "model.keras"))
            
            # メタデータの保存
            metadata = {
                "original_model": model_name,
                "converted_to": "768_dimensional_standard",
                "embedding_dimension": DEFAULT_EMBEDDING_DIM,
                "config": STANDARD_MODEL_CONFIG,
                "conversion_date": str(Path(__file__).stat().st_mtime)
            }
            
            with open(new_model_path / "conversion_metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)
            
            conversion_results[model_name] = {
                "status": "converted",
                "new_path": str(new_model_path),
                "metadata": metadata
            }
            
            logger.info(f"Successfully converted {model_name} to 768-dimensional standard")
            
        except Exception as error:
            logger.error(f"Failed to convert {model_name}: {error}")
            conversion_results[model_name] = {
                "status": "conversion_error",
                "error": str(error)
            }
    
    return conversion_results

def generate_tensorflowjs_models(
    validation_results: Dict[str, Dict], 
    conversion_results: Dict[str, Dict],
    models_dir: Path
) -> Dict[str, Dict]:
    """TensorFlow.js形式の生成"""
    logger.info("Generating TensorFlow.js models")
    
    js_results = {}
    converter = create_tensorflowjs_converter()
    
    # 768次元準拠のモデルを特定
    target_models = []
    
    for model_name, validation in validation_results.items():
        if validation["status"] == "success" and validation["is_768_compliant"]:
            target_models.append({
                "name": model_name,
                "path": models_dir / model_name,
                "source": "original"
            })
    
    for model_name, conversion in conversion_results.items():
        if conversion["status"] == "converted":
            target_models.append({
                "name": f"{model_name}_768",
                "path": Path(conversion["new_path"]),
                "source": "converted"
            })
    
    # 各モデルをTensorFlow.js形式に変換
    for model_info in target_models:
        model_name = model_info["name"]
        model_path = model_info["path"]
        
        logger.info(f"Converting {model_name} to TensorFlow.js")
        
        try:
            # モデルファイルの確認
            keras_files = list(model_path.glob("*.keras")) + list(model_path.glob("*.h5"))
            if not keras_files:
                logger.warning(f"No Keras model file found for {model_name}")
                continue
            
            model_file = keras_files[0]
            
            # 速度最適化版の変換
            speed_result = converter.convert_two_tower_model(
                str(model_file),
                f"{model_name}_speed",
                quantize=True,
                optimize_for="speed"
            )
            
            # サイズ最適化版の変換
            size_result = converter.convert_two_tower_model(
                str(model_file),
                f"{model_name}_size",
                quantize=True,
                optimize_for="size"
            )
            
            js_results[model_name] = {
                "status": "success",
                "speed_optimized": speed_result,
                "size_optimized": size_result,
                "source": model_info["source"]
            }
            
            logger.info(f"Successfully converted {model_name} to TensorFlow.js")
            
        except Exception as error:
            logger.error(f"TensorFlow.js conversion failed for {model_name}: {error}")
            js_results[model_name] = {
                "status": "js_conversion_error",
                "error": str(error)
            }
    
    return js_results

def create_deployment_package(js_results: Dict[str, Dict]) -> Dict[str, any]:
    """配信パッケージの作成"""
    logger.info("Creating deployment package")
    
    converter = create_tensorflowjs_converter()
    
    # 成功した変換のモデル名を取得
    successful_models = []
    for model_name, result in js_results.items():
        if result["status"] == "success":
            successful_models.extend([
                f"{model_name}_speed",
                f"{model_name}_size"
            ])
    
    if not successful_models:
        logger.warning("No successful TensorFlow.js models found for deployment")
        return {"status": "no_models", "models": []}
    
    # Web配信パッケージの作成
    package_result = converter.create_web_deployment_package(successful_models)
    
    return package_result

def main():
    """メイン実行関数"""
    parser = argparse.ArgumentParser(description="768-Dimensional Model Standardization")
    parser.add_argument("--models-dir", type=Path, default=Path("models"), 
                       help="Models directory path")
    parser.add_argument("--create-standard", action="store_true",
                       help="Create new 768-dimensional standard model")
    parser.add_argument("--convert-existing", action="store_true", 
                       help="Convert existing models to 768-dimensional")
    parser.add_argument("--generate-js", action="store_true",
                       help="Generate TensorFlow.js models")
    parser.add_argument("--create-package", action="store_true",
                       help="Create web deployment package")
    parser.add_argument("--all", action="store_true",
                       help="Run all operations")
    parser.add_argument("--output-dir", type=Path, default=Path("output"),
                       help="Output directory for results")
    
    args = parser.parse_args()
    
    if args.all:
        args.create_standard = True
        args.convert_existing = True
        args.generate_js = True
        args.create_package = True
    
    # 出力ディレクトリの作成
    args.output_dir.mkdir(exist_ok=True)
    
    results = {
        "timestamp": str(Path(__file__).stat().st_mtime),
        "operations": [],
        "summary": {}
    }
    
    try:
        # 1. 新しい768次元標準モデルの作成
        if args.create_standard:
            logger.info("Creating new 768-dimensional standard model")
            try:
                standard_model_path = create_768_standard_models()
                results["operations"].append({
                    "operation": "create_standard",
                    "status": "success",
                    "model_path": standard_model_path
                })
                logger.info(f"Standard model created at {standard_model_path}")
            except Exception as error:
                logger.error(f"Failed to create standard model: {error}")
                results["operations"].append({
                    "operation": "create_standard",
                    "status": "error",
                    "error": str(error)
                })
        
        # 2. 既存モデルの検証
        validation_results = validate_existing_models(args.models_dir)
        results["validation_results"] = validation_results
        results["operations"].append({
            "operation": "validation",
            "status": "success",
            "models_found": len(validation_results),
            "compliant_models": sum(1 for v in validation_results.values() 
                                  if v.get("is_768_compliant", False))
        })
        
        # 3. 既存モデルの768次元変換
        conversion_results = {}
        if args.convert_existing:
            conversion_results = convert_models_to_768(validation_results, args.models_dir)
            results["conversion_results"] = conversion_results
            results["operations"].append({
                "operation": "conversion",
                "status": "success",
                "converted_models": sum(1 for c in conversion_results.values() 
                                      if c.get("status") == "converted")
            })
        
        # 4. TensorFlow.js変換
        js_results = {}
        if args.generate_js:
            js_results = generate_tensorflowjs_models(
                validation_results, conversion_results, args.models_dir
            )
            results["tensorflowjs_results"] = js_results
            results["operations"].append({
                "operation": "tensorflowjs_conversion",
                "status": "success",
                "js_models_created": sum(1 for j in js_results.values() 
                                       if j.get("status") == "success")
            })
        
        # 5. Web配信パッケージの作成
        package_result = {}
        if args.create_package and js_results:
            package_result = create_deployment_package(js_results)
            results["package_result"] = package_result
            results["operations"].append({
                "operation": "create_package",
                "status": "success" if package_result.get("success") else "error",
                **package_result
            })
        
        # 結果の要約
        results["summary"] = {
            "total_models_processed": len(validation_results),
            "768_compliant_models": sum(1 for v in validation_results.values() 
                                      if v.get("is_768_compliant", False)),
            "converted_models": sum(1 for c in conversion_results.values() 
                                  if c.get("status") == "converted"),
            "tensorflowjs_models": sum(1 for j in js_results.values() 
                                     if j.get("status") == "success"),
            "deployment_package_created": package_result.get("success", False)
        }
        
        # 結果の保存
        with open(args.output_dir / "standardization_results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        # サマリーの表示
        print("\n" + "="*60)
        print("768-Dimensional Model Standardization Results")
        print("="*60)
        print(f"Total models processed: {results['summary']['total_models_processed']}")
        print(f"768-compliant models: {results['summary']['768_compliant_models']}")
        print(f"Models converted: {results['summary']['converted_models']}")
        print(f"TensorFlow.js models: {results['summary']['tensorflowjs_models']}")
        print(f"Deployment package: {'Created' if results['summary']['deployment_package_created'] else 'Not created'}")
        print("="*60)
        
        logger.info("Model standardization completed successfully")
        
    except Exception as error:
        logger.error(f"Model standardization failed: {error}")
        results["error"] = str(error)
        
        with open(args.output_dir / "standardization_error.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        sys.exit(1)

if __name__ == "__main__":
    main()