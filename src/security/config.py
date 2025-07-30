"""
Brain-Forge Security Configuration

Security configuration and constants for the Brain-Forge platform.
"""

import os
from datetime import timedelta
from typing import Any, Dict, Optional


class SecurityConfig:
    """Security configuration settings"""
    
    # Authentication settings
    JWT_SECRET_KEY: Optional[str] = os.getenv('BRAIN_FORGE_JWT_SECRET')
    JWT_ALGORITHM: str = 'HS256'
    JWT_EXPIRY_HOURS: int = 24
    
    # Password requirements
    MIN_PASSWORD_LENGTH: int = 12
    REQUIRE_UPPERCASE: bool = True
    REQUIRE_LOWERCASE: bool = True
    REQUIRE_NUMBERS: bool = True
    REQUIRE_SYMBOLS: bool = True
    
    # Session settings
    SESSION_TIMEOUT_MINUTES: int = 60
    MAX_CONCURRENT_SESSIONS: int = 5
    
    # Encryption settings
    ENCRYPTION_ALGORITHM: str = 'AES-256-GCM'
    KEY_ROTATION_DAYS: int = 90
    
    # HIPAA compliance settings
    AUDIT_LOG_RETENTION_DAYS: int = 2555  # 7 years
    DATA_RETENTION_DAYS: int = 2555  # 7 years
    REQUIRE_AUDIT_LOG: bool = True
    
    # Rate limiting
    API_RATE_LIMIT_PER_MINUTE: int = 100
    LOGIN_RATE_LIMIT_PER_MINUTE: int = 5
    FAILED_LOGIN_LOCKOUT_MINUTES: int = 15
    
    # Security headers
    SECURITY_HEADERS: Dict[str, str] = {
        'X-Content-Type-Options': 'nosniff',
        'X-Frame-Options': 'DENY',
        'X-XSS-Protection': '1; mode=block',
        'Strict-Transport-Security': 'max-age=31536000; includeSubDomains',
        'Content-Security-Policy': "default-src 'self'",
        'Referrer-Policy': 'strict-origin-when-cross-origin'
    }
    
    # HIPAA compliance requirements
    HIPAA_REQUIREMENTS: Dict[str, Any] = {
        'encryption_at_rest': True,
        'encryption_in_transit': True,
        'access_controls': True,
        'audit_logging': True,
        'data_backup': True,
        'incident_response': True,
        'user_training': True,
        'risk_assessment': True
    }
    
    @classmethod
    def get_jwt_expiry(cls) -> timedelta:
        """Get JWT token expiry as timedelta"""
        return timedelta(hours=cls.JWT_EXPIRY_HOURS)
    
    @classmethod
    def get_session_timeout(cls) -> timedelta:
        """Get session timeout as timedelta"""
        return timedelta(minutes=cls.SESSION_TIMEOUT_MINUTES)
    
    @classmethod
    def validate_password(cls, password: str) -> tuple[bool, str]:
        """Validate password against requirements"""
        if len(password) < cls.MIN_PASSWORD_LENGTH:
            return False, (f"Password must be at least "
                           f"{cls.MIN_PASSWORD_LENGTH} characters")
        
        if cls.REQUIRE_UPPERCASE and not any(c.isupper() for c in password):
            return False, "Password must contain at least one uppercase letter"
        
        if cls.REQUIRE_LOWERCASE and not any(c.islower() for c in password):
            return False, "Password must contain at least one lowercase letter"
        
        if cls.REQUIRE_NUMBERS and not any(c.isdigit() for c in password):
            return False, "Password must contain at least one number"
        
        symbols = "!@#$%^&*()_+-=[]{}|;:,.<>?"
        if cls.REQUIRE_SYMBOLS and not any(c in symbols for c in password):
            return False, "Password must contain at least one symbol"
        
        return True, "Password meets requirements"


# Security constants
ALLOWED_FILE_EXTENSIONS = {
    'data': ['.edf', '.bdf', '.cnt', '.vhdr', '.fif', '.mat', '.csv'],
    'config': ['.yaml', '.yml', '.json'],
    'export': ['.csv', '.json', '.mat', '.h5', '.hdf5']
}

SENSITIVE_DATA_FIELDS = {
    'patient_info',
    'personal_data',
    'brain_data',
    'medical_records',
    'biometric_data'
}

# Default security roles and permissions
DEFAULT_ROLE_PERMISSIONS = {
    'super_admin': ['read', 'write', 'delete', 'admin'],
    'admin': ['read', 'write', 'delete'],
    'researcher': ['read', 'write'],
    'clinician': ['read', 'write'],
    'technician': ['read'],
    'viewer': ['read']
}

# API endpoint security requirements
ENDPOINT_SECURITY = {
    '/api/v1/auth/login': {'auth': False, 'rate_limit': 'strict'},
    '/api/v1/auth/logout': {'auth': True, 'rate_limit': 'normal'},
    '/api/v1/users': {'auth': True, 'permission': 'admin'},
    '/api/v1/brain-data': {'auth': True, 'permission': 'read'},
    '/api/v1/acquisition': {'auth': True, 'permission': 'write'},
    '/api/v1/processing': {'auth': True, 'permission': 'write'},
    '/api/v1/transfer-learning': {'auth': True, 'permission': 'write'},
    '/api/v1/admin': {'auth': True, 'permission': 'admin'}
}
