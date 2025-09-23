"""
Model Deployer

モデルデプロイメントユーティリティ
"""

import os
import json
import shutil
from pathlib import Path
from typing import Dict, Any, Optional
import tempfile


class ModelDeployer:
    """モデルデプロイメントクラス"""
    
    def __init__(self):
        pass
    
    def create_web_package(
        self,
        model_trainer,
        output_path: str,
        include_preprocessors: bool = True,
        minify_assets: bool = True
    ) -> Dict[str, Any]:
        """Web配信パッケージ作成"""
        try:
            output_dir = Path(output_path)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # パッケージ構造作成
            directories = ['model', 'preprocessors', 'config']
            for directory in directories:
                (output_dir / directory).mkdir(exist_ok=True)
            
            # package.json作成
            package_metadata = {
                "name": "two-tower-model-package",
                "version": "1.0.0",
                "description": "Two-Tower recommendation model for web deployment",
                "main": "inference.js",
                "dependencies": {
                    "@tensorflow/tfjs": "^4.0.0",
                    "@tensorflow/tfjs-node": "^4.0.0"
                },
                "scripts": {
                    "test": "node test.js",
                    "serve": "node server.js"
                },
                "keywords": ["machine-learning", "tensorflow", "recommendation"],
                "author": "Adult Matching ML Team",
                "license": "MIT"
            }
            
            package_json_path = output_dir / "package.json"
            with open(package_json_path, 'w') as f:
                json.dump(package_metadata, f, indent=2)
            
            # inference.js作成
            inference_script = '''
/**
 * Two-Tower Model Inference Script
 * TensorFlow.jsを使用したクライアントサイド推論
 */

class TwoTowerInference {
    constructor() {
        this.model = null;
        this.userPreprocessor = null;
        this.itemPreprocessor = null;
    }
    
    async loadModel(modelPath = './model/model.json') {
        try {
            // モデル読み込み
            this.model = await tf.loadGraphModel(modelPath);
            
            // 前処理器読み込み
            this.userPreprocessor = await this.loadPreprocessor('./preprocessors/user_preprocessor.json');
            this.itemPreprocessor = await this.loadPreprocessor('./preprocessors/item_preprocessor.json');
            
            console.log('Two-Tower model loaded successfully');
            return true;
        } catch (error) {
            console.error('Model loading failed:', error);
            return false;
        }
    }
    
    async loadPreprocessor(path) {
        const response = await fetch(path);
        return await response.json();
    }
    
    preprocess(userData, itemData) {
        // ユーザー特徴量前処理
        const userFeatures = this.preprocessUserFeatures(userData);
        
        // アイテム特徴量前処理
        const itemFeatures = this.preprocessItemFeatures(itemData);
        
        return {
            userFeatures: tf.tensor2d([userFeatures]),
            itemFeatures: tf.tensor2d([itemFeatures])
        };
    }
    
    preprocessUserFeatures(userData) {
        // 年齢正規化
        const normalizedAge = (userData.age - 30) / 20;
        
        // カテゴリカル特徴量エンコーディング
        const genderEncoding = userData.gender === 'M' ? 1 : 0;
        
        // 基本特徴量
        return [normalizedAge, genderEncoding, ...userData.interests_encoding];
    }
    
    preprocessItemFeatures(itemData) {
        // 価格正規化
        const normalizedPrice = (itemData.price - 5000) / 3000;
        
        // ジャンルエンコーディング
        const genreEncoding = this.encodeGenre(itemData.genre);
        
        return [normalizedPrice, itemData.rating / 5.0, ...genreEncoding];
    }
    
    encodeGenre(genre) {
        const genres = ['Action', 'Romance', 'Comedy', 'Drama', 'Horror'];
        const encoding = new Array(genres.length).fill(0);
        const index = genres.indexOf(genre);
        if (index !== -1) {
            encoding[index] = 1;
        }
        return encoding;
    }
    
    async predict(userData, itemData) {
        if (!this.model) {
            throw new Error('Model not loaded. Call loadModel() first.');
        }
        
        const { userFeatures, itemFeatures } = this.preprocess(userData, itemData);
        
        try {
            // 予測実行
            const prediction = this.model.predict({
                user_features: userFeatures,
                item_features: itemFeatures
            });
            
            const result = await prediction.data();
            
            // テンソルクリーンアップ
            userFeatures.dispose();
            itemFeatures.dispose();
            prediction.dispose();
            
            return result[0];
        } catch (error) {
            console.error('Prediction failed:', error);
            throw error;
        }
    }
    
    postprocess(prediction) {
        // 予測値の後処理
        return {
            score: prediction,
            confidence: prediction > 0.5 ? 'high' : 'low',
            recommendation: prediction > 0.7 ? 'strongly_recommend' : 
                          prediction > 0.5 ? 'recommend' : 'not_recommend'
        };
    }
}

// グローバル公開
if (typeof module !== 'undefined' && module.exports) {
    module.exports = TwoTowerInference;
} else {
    window.TwoTowerInference = TwoTowerInference;
}
'''
            
            inference_js_path = output_dir / "inference.js"
            with open(inference_js_path, 'w') as f:
                f.write(inference_script)
            
            # 前処理器設定ファイル
            if include_preprocessors:
                user_preprocessor = {
                    "type": "user_features",
                    "age_normalization": {"mean": 30, "std": 20},
                    "gender_encoding": {"M": 1, "F": 0},
                    "categorical_features": ["prefecture", "occupation"],
                    "feature_order": ["age", "gender", "prefecture", "occupation", "interests"]
                }
                
                item_preprocessor = {
                    "type": "item_features", 
                    "price_normalization": {"mean": 5000, "std": 3000},
                    "rating_normalization": {"min": 1, "max": 5},
                    "genre_encoding": {
                        "Action": [1, 0, 0, 0, 0],
                        "Romance": [0, 1, 0, 0, 0],
                        "Comedy": [0, 0, 1, 0, 0],
                        "Drama": [0, 0, 0, 1, 0],
                        "Horror": [0, 0, 0, 0, 1]
                    }
                }
                
                with open(output_dir / "preprocessors" / "user_preprocessor.json", 'w') as f:
                    json.dump(user_preprocessor, f, indent=2)
                
                with open(output_dir / "preprocessors" / "item_preprocessor.json", 'w') as f:
                    json.dump(item_preprocessor, f, indent=2)
            
            # 設定ファイル
            config = {
                "model_version": "1.0.0",
                "input_shape": {
                    "user_features": [-1, 10],
                    "item_features": [-1, 8]
                },
                "output_shape": [1],
                "prediction_threshold": 0.5,
                "preprocessing_required": include_preprocessors,
                "created_at": "2025-09-16"
            }
            
            with open(output_dir / "config" / "model_config.json", 'w') as f:
                json.dump(config, f, indent=2)
            
            # README.md作成
            readme_content = '''# Two-Tower Model Web Package

## Overview
This package contains a Two-Tower recommendation model optimized for web deployment using TensorFlow.js.

## Usage

```javascript
// Initialize model
const inference = new TwoTowerInference();

// Load model
await inference.loadModel('./model/model.json');

// Make prediction
const userData = {
    age: 25,
    gender: 'M',
    interests_encoding: [1, 0, 1]
};

const itemData = {
    price: 4500,
    rating: 4.2,
    genre: 'Action'
};

const prediction = await inference.predict(userData, itemData);
console.log('Prediction score:', prediction);
```

## Files Structure
- `model/`: TensorFlow.js model files
- `preprocessors/`: Feature preprocessing configurations
- `config/`: Model configuration
- `inference.js`: Main inference script
- `package.json`: Package metadata

## Requirements
- TensorFlow.js ^4.0.0
- Modern web browser with ES6 support
'''
            
            readme_path = output_dir / "README.md"
            with open(readme_path, 'w') as f:
                f.write(readme_content)
            
            return {
                'success': True,
                'output_path': str(output_path),
                'files_created': [
                    'package.json', 'inference.js', 'README.md',
                    'config/model_config.json'
                ] + (['preprocessors/user_preprocessor.json', 'preprocessors/item_preprocessor.json'] if include_preprocessors else [])
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }