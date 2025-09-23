/**
 * Backend 統合設定ファイル
 * 
 * 各モジュール間で共有する設定を一元管理
 */

export interface BackendConfig {
  edge_functions: EdgeFunctionConfig;
  ml_pipeline: MLPipelineConfig;
  data_processing: DataProcessingConfig;
  database: DatabaseConfig;
  external_apis: ExternalAPIConfig;
  monitoring: MonitoringConfig;
  security: SecurityConfig;
}

export interface EdgeFunctionConfig {
  cors: {
    allowed_origins: string[];
    allowed_headers: string[];
    allowed_methods: string[];
  };
  rate_limiting: {
    authenticated_users: number;  // requests per minute
    anonymous_users: number;
    admin_users: number;
  };
  caching: {
    user_embeddings_ttl: number;  // seconds
    recommendations_ttl: number;
    search_results_ttl: number;
  };
  timeout: {
    default: number;  // milliseconds
    ml_inference: number;
    database_query: number;
  };
}

export interface MLPipelineConfig {
  model_params: {
    embedding_dimension: number;
    two_tower_hidden_units: number[];
    learning_rate: number;
    batch_size: number;
    epochs: number;
  };
  inference: {
    similarity_threshold: number;
    diversity_weight: number;
    max_recommendations: number;
    embedding_cache_size: number;
  };
  training: {
    validation_split: number;
    early_stopping_patience: number;
    model_checkpoint_freq: number;
  };
  data: {
    min_user_likes: number;
    min_video_likes: number;
    feature_extraction_batch_size: number;
  };
}

export interface DataProcessingConfig {
  scraping: {
    request_delay_ms: number;
    max_concurrent_requests: number;
    retry_attempts: number;
    user_agent: string;
  };
  cleaning: {
    min_title_length: number;
    max_title_length: number;
    min_description_length: number;
    duplicate_threshold: number;
  };
  embedding: {
    batch_size: number;
    vector_dimension: number;
    similarity_index_type: string;
  };
  validation: {
    required_fields: string[];
    price_range: { min: number; max: number };
    image_url_validation: boolean;
  };
}

export interface DatabaseConfig {
  connection: {
    pool_size: number;
    max_connections: number;
    idle_timeout: number;
    query_timeout: number;
  };
  pgvector: {
    dimension: number;
    distance_function: 'cosine' | 'euclidean' | 'inner_product';
    index_type: 'ivfflat' | 'hnsw';
    index_params: {
      lists?: number;      // for ivfflat
      m?: number;          // for hnsw
      ef_construction?: number;  // for hnsw
    };
  };
  performance: {
    enable_query_caching: boolean;
    cache_ttl: number;
    enable_connection_pooling: boolean;
    log_slow_queries: boolean;
    slow_query_threshold_ms: number;
  };
}

export interface ExternalAPIConfig {
  dmm: {
    api_id: string;
    affiliate_id: string;
    base_url: string;
    rate_limit_per_second: number;
    timeout_ms: number;
    retry_attempts: number;
  };
  other_apis: {
    // 将来的な外部API設定
  };
}

export interface MonitoringConfig {
  logging: {
    level: 'DEBUG' | 'INFO' | 'WARN' | 'ERROR';
    enable_performance_logging: boolean;
    enable_error_tracking: boolean;
    log_rotation: {
      max_files: number;
      max_size_mb: number;
    };
  };
  metrics: {
    enable_performance_metrics: boolean;
    enable_business_metrics: boolean;
    metrics_collection_interval_ms: number;
    dashboard_update_interval_ms: number;
  };
  alerting: {
    enable_error_alerts: boolean;
    enable_performance_alerts: boolean;
    error_threshold: number;
    response_time_threshold_ms: number;
  };
}

export interface SecurityConfig {
  authentication: {
    jwt_expiry_minutes: number;
    refresh_token_expiry_days: number;
    password_min_length: number;
    require_email_verification: boolean;
  };
  authorization: {
    enable_rbac: boolean;
    admin_email_domains: string[];
    rate_limit_by_ip: boolean;
    enable_captcha: boolean;
  };
  data_protection: {
    enable_data_encryption: boolean;
    enable_audit_logging: boolean;
    data_retention_days: number;
    enable_gdpr_compliance: boolean;
  };
}

// ============================================================================
// 環境別設定
// ============================================================================

const developmentConfig: BackendConfig = {
  edge_functions: {
    cors: {
      allowed_origins: ['http://localhost:3000', 'http://localhost:3001'],
      allowed_headers: ['authorization', 'x-client-info', 'apikey', 'content-type'],
      allowed_methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS']
    },
    rate_limiting: {
      authenticated_users: 120,
      anonymous_users: 60,
      admin_users: 300
    },
    caching: {
      user_embeddings_ttl: 300,  // 5分
      recommendations_ttl: 600,  // 10分
      search_results_ttl: 180    // 3分
    },
    timeout: {
      default: 30000,
      ml_inference: 60000,
      database_query: 15000
    }
  },
  ml_pipeline: {
    model_params: {
      embedding_dimension: 768,
      two_tower_hidden_units: [512, 256, 128],
      learning_rate: 0.001,
      batch_size: 128,
      epochs: 50
    },
    inference: {
      similarity_threshold: 0.1,
      diversity_weight: 0.3,
      max_recommendations: 50,
      embedding_cache_size: 10000
    },
    training: {
      validation_split: 0.2,
      early_stopping_patience: 5,
      model_checkpoint_freq: 5
    },
    data: {
      min_user_likes: 5,
      min_video_likes: 10,
      feature_extraction_batch_size: 1000
    }
  },
  data_processing: {
    scraping: {
      request_delay_ms: 1000,
      max_concurrent_requests: 5,
      retry_attempts: 3,
      user_agent: 'Mozilla/5.0 (compatible; AdultMatchingBot/1.0)'
    },
    cleaning: {
      min_title_length: 10,
      max_title_length: 200,
      min_description_length: 20,
      duplicate_threshold: 0.95
    },
    embedding: {
      batch_size: 100,
      vector_dimension: 768,
      similarity_index_type: 'cosine'
    },
    validation: {
      required_fields: ['title', 'description', 'maker', 'genre', 'price'],
      price_range: { min: 0, max: 50000 },
      image_url_validation: true
    }
  },
  database: {
    connection: {
      pool_size: 20,
      max_connections: 100,
      idle_timeout: 30000,
      query_timeout: 15000
    },
    pgvector: {
      dimension: 768,
      distance_function: 'cosine',
      index_type: 'ivfflat',
      index_params: {
        lists: 100
      }
    },
    performance: {
      enable_query_caching: true,
      cache_ttl: 300,
      enable_connection_pooling: true,
      log_slow_queries: true,
      slow_query_threshold_ms: 1000
    }
  },
  external_apis: {
    dmm: {
      api_id: process.env.DMM_API_ID || '',
      affiliate_id: process.env.DMM_AFFILIATE_ID || '',
      base_url: 'https://api.dmm.com/affiliate/v3/',
      rate_limit_per_second: 1,
      timeout_ms: 30000,
      retry_attempts: 3
    },
    other_apis: {}
  },
  monitoring: {
    logging: {
      level: 'DEBUG',
      enable_performance_logging: true,
      enable_error_tracking: true,
      log_rotation: {
        max_files: 10,
        max_size_mb: 100
      }
    },
    metrics: {
      enable_performance_metrics: true,
      enable_business_metrics: true,
      metrics_collection_interval_ms: 60000,
      dashboard_update_interval_ms: 30000
    },
    alerting: {
      enable_error_alerts: true,
      enable_performance_alerts: true,
      error_threshold: 10,
      response_time_threshold_ms: 5000
    }
  },
  security: {
    authentication: {
      jwt_expiry_minutes: 60,
      refresh_token_expiry_days: 7,
      password_min_length: 8,
      require_email_verification: false
    },
    authorization: {
      enable_rbac: true,
      admin_email_domains: ['admin.localhost'],
      rate_limit_by_ip: true,
      enable_captcha: false
    },
    data_protection: {
      enable_data_encryption: false,
      enable_audit_logging: true,
      data_retention_days: 90,
      enable_gdpr_compliance: true
    }
  }
};

const productionConfig: BackendConfig = {
  ...developmentConfig,
  edge_functions: {
    ...developmentConfig.edge_functions,
    cors: {
      allowed_origins: ['https://your-domain.com'],
      allowed_headers: ['authorization', 'x-client-info', 'apikey', 'content-type'],
      allowed_methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS']
    },
    rate_limiting: {
      authenticated_users: 300,
      anonymous_users: 100,
      admin_users: 1000
    },
    caching: {
      user_embeddings_ttl: 1800,  // 30分
      recommendations_ttl: 3600,  // 1時間
      search_results_ttl: 900     // 15分
    }
  },
  database: {
    ...developmentConfig.database,
    connection: {
      pool_size: 100,
      max_connections: 500,
      idle_timeout: 60000,
      query_timeout: 30000
    },
    pgvector: {
      dimension: 768,
      distance_function: 'cosine',
      index_type: 'hnsw',
      index_params: {
        m: 16,
        ef_construction: 200
      }
    }
  },
  monitoring: {
    ...developmentConfig.monitoring,
    logging: {
      ...developmentConfig.monitoring.logging,
      level: 'INFO'
    }
  },
  security: {
    ...developmentConfig.security,
    authentication: {
      ...developmentConfig.security.authentication,
      require_email_verification: true
    },
    authorization: {
      ...developmentConfig.security.authorization,
      enable_captcha: true
    },
    data_protection: {
      ...developmentConfig.security.data_protection,
      enable_data_encryption: true,
      data_retention_days: 365
    }
  }
};

// ============================================================================
// 設定エクスポート
// ============================================================================

export function getBackendConfig(): BackendConfig {
  const environment = Deno.env.get('ENVIRONMENT') || 'development';
  
  switch (environment) {
    case 'production':
      return productionConfig;
    case 'development':
    default:
      return developmentConfig;
  }
}

export default getBackendConfig();