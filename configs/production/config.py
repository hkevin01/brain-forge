"""
Production Configuration for Brain-Forge Platform

This module contains production-specific configuration settings.
"""

import os
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class ProductionConfig:
    """Production environment configuration"""
    
    # Environment
    ENVIRONMENT: str = "production"
    DEBUG: bool = False
    TESTING: bool = False
    
    # API Configuration
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_WORKERS: int = 4
    API_TIMEOUT: int = 300
    
    # Database Configuration
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL",
        "postgresql://brain_forge_user:password@postgres-service:5432/"
        "brain_forge"
    )
    DATABASE_POOL_SIZE: int = 20
    DATABASE_MAX_OVERFLOW: int = 30
    DATABASE_POOL_TIMEOUT: int = 30
    DATABASE_POOL_RECYCLE: int = 3600
    
    # Redis Configuration
    REDIS_URL: str = os.getenv(
        "REDIS_URL",
        "redis://:password@redis-service:6379"
    )
    REDIS_MAX_CONNECTIONS: int = 50
    REDIS_SOCKET_TIMEOUT: int = 5
    REDIS_SOCKET_CONNECT_TIMEOUT: int = 5
    
    # Security Configuration
    JWT_SECRET_KEY: str = os.getenv(
        "JWT_SECRET_KEY", "change-me-in-production"
    )
    JWT_ALGORITHM: str = "HS256"
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = 60
    JWT_REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    ENCRYPTION_KEY: str = os.getenv(
        "ENCRYPTION_KEY", "change-me-in-production"
    )
    
    # CORS Configuration
    CORS_ORIGINS: List[str] = [
        "https://brain-forge.example.com",
        "https://api.brain-forge.example.com",
        "https://app.brain-forge.example.com"
    ]
    CORS_ALLOW_CREDENTIALS: bool = True
    CORS_ALLOW_METHODS: List[str] = ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
    CORS_ALLOW_HEADERS: List[str] = ["*"]
    
    # Rate Limiting
    RATE_LIMIT_ENABLED: bool = True
    API_RATE_LIMIT_PER_MINUTE: int = 100
    LOGIN_RATE_LIMIT_PER_MINUTE: int = 5
    FAILED_LOGIN_LOCKOUT_MINUTES: int = 30
    
    # Session Configuration
    SESSION_TIMEOUT_MINUTES: int = 60
    MAX_LOGIN_ATTEMPTS: int = 5
    
    # Logging Configuration
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "json"
    LOG_FILE_MAX_SIZE: int = 100 * 1024 * 1024  # 100MB
    LOG_FILE_BACKUP_COUNT: int = 10
    AUDIT_LOG_RETENTION_DAYS: int = 2555  # 7 years for HIPAA
    
    # Hardware Configuration
    OMP_DEVICE_COUNT: int = int(os.getenv("OMP_DEVICE_COUNT", "4"))
    KERNEL_HELMET_COUNT: int = int(os.getenv("KERNEL_HELMET_COUNT", "2"))
    ACCELEROMETER_SAMPLING_RATE: int = int(
        os.getenv("ACCELEROMETER_SAMPLING_RATE", "1000")
    )
    
    # Processing Configuration
    PROCESSING_BUFFER_SIZE: int = int(
        os.getenv("PROCESSING_BUFFER_SIZE", "1024")
    )
    PROCESSING_OVERLAP: float = float(os.getenv("PROCESSING_OVERLAP", "0.5"))
    GPU_ENABLED: bool = os.getenv("GPU_ENABLED", "false").lower() == "true"
    MAX_PROCESSING_THREADS: int = 8
    
    # WebSocket Configuration
    WEBSOCKET_MAX_CONNECTIONS: int = int(
        os.getenv("WEBSOCKET_MAX_CONNECTIONS", "200")
    )
    WEBSOCKET_PING_INTERVAL: int = 30
    WEBSOCKET_PING_TIMEOUT: int = 10
    
    # File Storage Configuration
    DATA_STORAGE_PATH: str = "/home/brain-forge/data"
    RESULTS_STORAGE_PATH: str = "/home/brain-forge/results"
    LOGS_STORAGE_PATH: str = "/home/brain-forge/logs"
    MAX_UPLOAD_SIZE: int = 100 * 1024 * 1024  # 100MB
    
    # Monitoring and Health Checks
    HEALTH_CHECK_INTERVAL: int = 30
    METRICS_ENABLED: bool = True
    PROMETHEUS_METRICS_PORT: int = 9090
    
    # External Service Configuration
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    EXTERNAL_API_TIMEOUT: int = 30
    EXTERNAL_API_RETRIES: int = 3
    
    # Backup Configuration
    BACKUP_ENABLED: bool = True
    BACKUP_INTERVAL_HOURS: int = 24
    BACKUP_RETENTION_DAYS: int = 30
    
    # Performance Configuration
    RESPONSE_CACHE_TTL: int = 300  # 5 minutes
    DATABASE_QUERY_TIMEOUT: int = 30
    MAX_CONCURRENT_REQUESTS: int = 1000
    
    # Security Headers
    SECURITY_HEADERS: Dict[str, str] = {
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "X-XSS-Protection": "1; mode=block",
        "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
        "Referrer-Policy": "strict-origin-when-cross-origin",
        "Content-Security-Policy": (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data: https:; "
            "font-src 'self'; "
            "connect-src 'self' wss: ws:;"
        )
    }
    
    # Feature Flags
    ENABLE_API_DOCS: bool = False  # Disabled in production
    ENABLE_WEBSOCKET_COMPRESSION: bool = True
    ENABLE_RESPONSE_COMPRESSION: bool = True
    ENABLE_REQUEST_LOGGING: bool = True
    ENABLE_PERFORMANCE_MONITORING: bool = True
    
    @classmethod
    def validate(cls) -> bool:
        """Validate production configuration"""
        errors = []
        
        # Check required environment variables
        required_vars = [
            "DATABASE_URL",
            "REDIS_URL",
            "JWT_SECRET_KEY",
            "ENCRYPTION_KEY"
        ]
        
        for var in required_vars:
            if not os.getenv(var):
                errors.append(f"Missing required environment variable: {var}")
        
        # Validate security settings
        if cls.JWT_SECRET_KEY == "change-me-in-production":
            errors.append("JWT_SECRET_KEY must be changed in production")
        
        if cls.ENCRYPTION_KEY == "change-me-in-production":
            errors.append("ENCRYPTION_KEY must be changed in production")
        
        if cls.DEBUG:
            errors.append("DEBUG must be False in production")
        
        if cls.ENABLE_API_DOCS:
            errors.append("API docs should be disabled in production")
        
        if errors:
            print("Production configuration validation errors:")
            for error in errors:
                print(f"  - {error}")
            return False
        
        return True
    
    @classmethod
    def load_from_env(cls) -> "ProductionConfig":
        """Load configuration from environment variables"""
        config = cls()
        
        # Override with environment variables
        env_overrides = {
            "API_HOST": os.getenv("API_HOST"),
            "API_PORT": os.getenv("API_PORT"),
            "LOG_LEVEL": os.getenv("LOG_LEVEL"),
            "DEBUG": os.getenv("DEBUG", "false").lower() == "true",
            "CORS_ORIGINS": (
                os.getenv("CORS_ORIGINS", "").split(",")
                if os.getenv("CORS_ORIGINS")
                else config.CORS_ORIGINS
            ),
        }
        
        for key, value in env_overrides.items():
            if value is not None:
                setattr(config, key, value)
        
        return config


# Global production configuration instance
production_config = ProductionConfig.load_from_env()
