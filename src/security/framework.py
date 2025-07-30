"""
Brain-Forge Security Framework

Comprehensive security implementation for the Brain-Forge BCI platform.
Provides authentication, authorization, encryption, and HIPAA compliance.
"""

import base64
import hashlib
import json
import logging
import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

# Encryption imports
try:
    from cryptography.fernet import Fernet
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False

# JWT imports
try:
    import jwt
    JWT_AVAILABLE = True
except ImportError:
    JWT_AVAILABLE = False

from core.config import Config
from core.exceptions import BrainForgeError
from core.logger import get_logger


class SecurityError(BrainForgeError):
    """Security-related errors"""
    pass


class AuthenticationError(SecurityError):
    """Authentication failures"""
    pass


class AuthorizationError(SecurityError):
    """Authorization failures"""
    pass


class EncryptionError(SecurityError):
    """Encryption/decryption failures"""
    pass


class UserRole(Enum):
    """User roles for RBAC"""
    SUPER_ADMIN = "super_admin"
    ADMIN = "admin"
    RESEARCHER = "researcher"
    CLINICIAN = "clinician"
    TECHNICIAN = "technician"
    VIEWER = "viewer"


class PermissionLevel(Enum):
    """Permission levels for data access"""
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    ADMIN = "admin"


@dataclass
class User:
    """User account representation"""
    user_id: str
    username: str
    email: str
    role: UserRole
    permissions: List[PermissionLevel]
    created_at: datetime
    last_login: Optional[datetime] = None
    is_active: bool = True
    password_hash: Optional[str] = None
    
    def has_permission(self, permission: PermissionLevel) -> bool:
        """Check if user has specific permission"""
        return (permission in self.permissions or
                PermissionLevel.ADMIN in self.permissions)


@dataclass
class SecurityContext:
    """Security context for requests"""
    user: Optional[User] = None
    session_id: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()


class EncryptionManager:
    """Handles encryption and decryption of sensitive data"""
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize encryption manager"""
        self.config = config or Config()
        self.logger = get_logger(f"{__name__}.EncryptionManager")
        
        if not CRYPTO_AVAILABLE:
            self.logger.error("Cryptography library not available")
            raise EncryptionError("Cryptography dependencies not installed")
        
        # Initialize encryption keys
        self._master_key = self._load_or_generate_master_key()
        self._fernet = Fernet(self._master_key)
        
        self.logger.info("Encryption manager initialized")
    
    def _load_or_generate_master_key(self) -> bytes:
        """Load existing master key or generate new one"""
        try:
            # In production, load from secure key management service
            # For development, generate a key
            key_file = self.config.get_data_dir() / "master.key"
            
            if key_file.exists():
                with open(key_file, "rb") as f:
                    key = f.read()
                self.logger.info("Master key loaded from file")
            else:
                # Generate new key
                key = Fernet.generate_key()
                key_file.parent.mkdir(parents=True, exist_ok=True)
                with open(key_file, "wb") as f:
                    f.write(key)
                self.logger.info("New master key generated and saved")
            
            return key
            
        except Exception as e:
            self.logger.error(f"Failed to initialize master key: {e}")
            raise EncryptionError(f"Master key initialization failed: {e}")
    
    def encrypt_data(self, data: bytes) -> bytes:
        """Encrypt data using AES-256"""
        try:
            return self._fernet.encrypt(data)
        except Exception as e:
            self.logger.error(f"Encryption failed: {e}")
            raise EncryptionError(f"Data encryption failed: {e}")
    
    def decrypt_data(self, encrypted_data: bytes) -> bytes:
        """Decrypt data"""
        try:
            return self._fernet.decrypt(encrypted_data)
        except Exception as e:
            self.logger.error(f"Decryption failed: {e}")
            raise EncryptionError(f"Data decryption failed: {e}")
    
    def encrypt_brain_data(self, brain_data: Dict[str, Any]) -> bytes:
        """Encrypt brain data with specialized handling"""
        try:
            # Convert to JSON and encrypt
            json_data = json.dumps(brain_data, default=str)
            return self.encrypt_data(json_data.encode('utf-8'))
            
        except Exception as e:
            self.logger.error(f"Brain data encryption failed: {e}")
            raise EncryptionError(f"Brain data encryption failed: {e}")
    
    def decrypt_brain_data(self, encrypted_data: bytes) -> Dict[str, Any]:
        """Decrypt brain data"""
        try:
            decrypted_bytes = self.decrypt_data(encrypted_data)
            json_str = decrypted_bytes.decode('utf-8')
            return json.loads(json_str)
            
        except Exception as e:
            self.logger.error(f"Brain data decryption failed: {e}")
            raise EncryptionError(f"Brain data decryption failed: {e}")


class AuthenticationManager:
    """Handles user authentication and JWT tokens"""
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize authentication manager"""
        self.config = config or Config()
        self.logger = get_logger(f"{__name__}.AuthenticationManager")
        
        if not JWT_AVAILABLE:
            self.logger.error("PyJWT library not available")
            raise AuthenticationError("JWT dependencies not installed")
        
        # JWT configuration
        self.jwt_secret = self._get_jwt_secret()
        self.jwt_algorithm = "HS256"
        self.token_expiry = timedelta(hours=24)
        
        # User storage (in production, use database)
        self._users: Dict[str, User] = {}
        self._sessions: Dict[str, SecurityContext] = {}
        
        # Create default admin user
        self._create_default_users()
        
        self.logger.info("Authentication manager initialized")
    
    def _get_jwt_secret(self) -> str:
        """Get JWT secret key"""
        # In production, load from environment or secure storage
        secret_file = self.config.get_data_dir() / "jwt_secret.key"
        
        if secret_file.exists():
            with open(secret_file, "r") as f:
                secret = f.read().strip()
        else:
            # Generate new secret
            secret = secrets.token_urlsafe(64)
            secret_file.parent.mkdir(parents=True, exist_ok=True)
            with open(secret_file, "w") as f:
                f.write(secret)
        
        return secret
    
    def _create_default_users(self):
        """Create default users for development"""
        # Default admin user
        admin_user = User(
            user_id="admin_001",
            username="admin",
            email="admin@brain-forge.dev",
            role=UserRole.SUPER_ADMIN,
            permissions=[PermissionLevel.ADMIN],
            created_at=datetime.utcnow(),
            password_hash=self._hash_password("admin_password_123")
        )
        
        # Default researcher user
        researcher_user = User(
            user_id="researcher_001",
            username="researcher",
            email="researcher@brain-forge.dev",
            role=UserRole.RESEARCHER,
            permissions=[PermissionLevel.READ, PermissionLevel.WRITE],
            created_at=datetime.utcnow(),
            password_hash=self._hash_password("researcher_password_123")
        )
        
        self._users["admin"] = admin_user
        self._users["researcher"] = researcher_user
        
        self.logger.info(f"Created {len(self._users)} default users")
    
    def _hash_password(self, password: str) -> str:
        """Hash password using PBKDF2"""
        salt = secrets.token_bytes(32)
        pwdhash = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, 100000)
        return base64.b64encode(salt + pwdhash).decode('ascii')
    
    def _verify_password(self, password: str, hashed: str) -> bool:
        """Verify password against hash"""
        try:
            decoded = base64.b64decode(hashed.encode('ascii'))
            salt = decoded[:32]
            stored_hash = decoded[32:]
            pwdhash = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, 100000)
            return pwdhash == stored_hash
        except:
            return False
    
    def authenticate_user(self, username: str, password: str) -> Optional[User]:
        """Authenticate user credentials"""
        try:
            user = self._users.get(username)
            if not user or not user.is_active:
                self.logger.warning(f"Authentication failed for user: {username}")
                return None
            
            if not self._verify_password(password, user.password_hash):
                self.logger.warning(f"Password verification failed for user: {username}")
                return None
            
            # Update last login
            user.last_login = datetime.utcnow()
            self.logger.info(f"User authenticated successfully: {username}")
            return user
            
        except Exception as e:
            self.logger.error(f"Authentication error: {e}")
            return None
    
    def create_token(self, user: User) -> str:
        """Create JWT token for authenticated user"""
        try:
            payload = {
                'user_id': user.user_id,
                'username': user.username,
                'role': user.role.value,
                'permissions': [p.value for p in user.permissions],
                'exp': datetime.utcnow() + self.token_expiry,
                'iat': datetime.utcnow()
            }
            
            token = jwt.encode(payload, self.jwt_secret, algorithm=self.jwt_algorithm)
            self.logger.info(f"JWT token created for user: {user.username}")
            return token
            
        except Exception as e:
            self.logger.error(f"Token creation failed: {e}")
            raise AuthenticationError(f"Token creation failed: {e}")
    
    def verify_token(self, token: str) -> Optional[User]:
        """Verify JWT token and return user"""
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=[self.jwt_algorithm])
            
            username = payload.get('username')
            user = self._users.get(username)
            
            if not user or not user.is_active:
                return None
            
            self.logger.debug(f"Token verified for user: {username}")
            return user
            
        except jwt.ExpiredSignatureError:
            self.logger.warning("Token has expired")
            return None
        except jwt.InvalidTokenError as e:
            self.logger.warning(f"Invalid token: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Token verification error: {e}")
            return None


class AuthorizationManager:
    """Handles user authorization and permissions"""
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize authorization manager"""
        self.config = config or Config()
        self.logger = get_logger(f"{__name__}.AuthorizationManager")
        
        # Resource permissions mapping
        self.resource_permissions = {
            'brain_data': [PermissionLevel.READ, PermissionLevel.WRITE, PermissionLevel.DELETE],
            'user_management': [PermissionLevel.ADMIN],
            'system_config': [PermissionLevel.ADMIN],
            'api_access': [PermissionLevel.READ],
            'data_export': [PermissionLevel.WRITE],
        }
        
        self.logger.info("Authorization manager initialized")
    
    def check_permission(self, user: User, resource: str, permission: PermissionLevel) -> bool:
        """Check if user has permission for resource"""
        try:
            # Super admin has all permissions
            if user.role == UserRole.SUPER_ADMIN:
                return True
            
            # Check if resource exists and permission is valid
            if resource not in self.resource_permissions:
                self.logger.warning(f"Unknown resource: {resource}")
                return False
            
            if permission not in self.resource_permissions[resource]:
                self.logger.warning(f"Invalid permission {permission} for resource {resource}")
                return False
            
            # Check user permissions
            has_permission = user.has_permission(permission)
            
            if not has_permission:
                self.logger.warning(f"Permission denied: {user.username} -> {resource}:{permission}")
            
            return has_permission
            
        except Exception as e:
            self.logger.error(f"Permission check error: {e}")
            return False


class AuditLogger:
    """Handles security audit logging for HIPAA compliance"""
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize audit logger"""
        self.config = config or Config()
        self.logger = get_logger(f"{__name__}.AuditLogger")
        
        # Audit log file
        self.audit_log_file = self.config.get_logs_dir() / "security_audit.log"
        self.audit_log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Configure audit logger
        self.audit_logger = logging.getLogger("security_audit")
        self.audit_logger.setLevel(logging.INFO)
        
        # File handler for audit logs
        handler = logging.FileHandler(self.audit_log_file)
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S UTC'
        )
        handler.setFormatter(formatter)
        self.audit_logger.addHandler(handler)
        
        self.logger.info("Audit logger initialized")
    
    def log_authentication(self, username: str, success: bool, ip_address: str = None):
        """Log authentication attempt"""
        status = "SUCCESS" if success else "FAILURE"
        self.audit_logger.info(
            f"AUTH_{status} | User: {username} | IP: {ip_address or 'unknown'}"
        )
    
    def log_data_access(self, user: User, resource: str, action: str, details: str = None):
        """Log data access events"""
        self.audit_logger.info(
            f"DATA_ACCESS | User: {user.username} | Resource: {resource} | "
            f"Action: {action} | Details: {details or 'none'}"
        )
    
    def log_security_event(self, event_type: str, user: User, details: str):
        """Log security events"""
        self.audit_logger.info(
            f"SECURITY_EVENT | Type: {event_type} | User: {user.username} | "
            f"Details: {details}"
        )
    
    def log_permission_check(self, user: User, resource: str, permission: str, granted: bool):
        """Log permission checks"""
        status = "GRANTED" if granted else "DENIED"
        self.audit_logger.info(
            f"PERMISSION_{status} | User: {user.username} | Resource: {resource} | "
            f"Permission: {permission}"
        )


class SecurityManager:
    """Main security manager coordinating all security components"""
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize security manager"""
        self.config = config or Config()
        self.logger = get_logger(f"{__name__}.SecurityManager")
        
        # Initialize security components
        try:
            self.encryption = EncryptionManager(config)
            self.authentication = AuthenticationManager(config)
            self.authorization = AuthorizationManager(config)
            self.audit = AuditLogger(config)
            
            self.logger.info("Security manager initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Security manager initialization failed: {e}")
            raise SecurityError(f"Security initialization failed: {e}")
    
    def authenticate_request(self, token: str, ip_address: str = None) -> SecurityContext:
        """Authenticate request and create security context"""
        try:
            user = self.authentication.verify_token(token)
            
            if not user:
                self.audit.log_authentication("unknown", False, ip_address)
                raise AuthenticationError("Invalid or expired token")
            
            # Create security context
            context = SecurityContext(
                user=user,
                session_id=secrets.token_urlsafe(32),
                ip_address=ip_address,
                timestamp=datetime.utcnow()
            )
            
            self.audit.log_authentication(user.username, True, ip_address)
            return context
            
        except Exception as e:
            self.logger.error(f"Request authentication failed: {e}")
            raise AuthenticationError(f"Authentication failed: {e}")
    
    def authorize_action(self, context: SecurityContext, resource: str, 
                        permission: PermissionLevel) -> bool:
        """Authorize user action"""
        try:
            if not context.user:
                return False
            
            granted = self.authorization.check_permission(
                context.user, resource, permission
            )
            
            self.audit.log_permission_check(
                context.user, resource, permission.value, granted
            )
            
            if not granted:
                raise AuthorizationError(
                    f"Insufficient permissions for {resource}:{permission.value}"
                )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Authorization failed: {e}")
            return False
    
    def encrypt_sensitive_data(self, data: Dict[str, Any]) -> bytes:
        """Encrypt sensitive data"""
        return self.encryption.encrypt_brain_data(data)
    
    def decrypt_sensitive_data(self, encrypted_data: bytes) -> Dict[str, Any]:
        """Decrypt sensitive data"""
        return self.encryption.decrypt_brain_data(encrypted_data)


# Security utilities
def require_authentication(security_manager: SecurityManager):
    """Decorator for authentication requirement"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Extract token from request (implementation depends on framework)
            token = kwargs.get('token') or kwargs.get('authorization')
            if not token:
                raise AuthenticationError("Authentication token required")
            
            context = security_manager.authenticate_request(token)
            kwargs['security_context'] = context
            
            return func(*args, **kwargs)
        return wrapper
    return decorator


def require_permission(security_manager: SecurityManager, resource: str, 
                      permission: PermissionLevel):
    """Decorator for permission requirement"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            context = kwargs.get('security_context')
            if not context:
                raise AuthenticationError("Security context required")
            
            if not security_manager.authorize_action(context, resource, permission):
                raise AuthorizationError(f"Permission denied: {resource}:{permission.value}")
            
            return func(*args, **kwargs)
        return wrapper
    return decorator


# Initialize global security manager
def create_security_manager(config: Optional[Config] = None) -> SecurityManager:
    """Create security manager instance"""
    return SecurityManager(config)


if __name__ == "__main__":
    # Demo security functionality
    print("🔐 Brain-Forge Security Framework Demo")
    
    try:
        # Initialize security
        security = create_security_manager()
        
        # Test authentication
        user = security.authentication.authenticate_user("admin", "admin_password_123")
        if user:
            print(f"✅ User authenticated: {user.username} ({user.role.value})")
            
            # Create token
            token = security.authentication.create_token(user)
            print(f"✅ Token created: {token[:50]}...")
            
            # Test encryption
            test_data = {"patient_id": "P001", "brain_activity": [1, 2, 3, 4, 5]}
            encrypted = security.encrypt_sensitive_data(test_data)
            print(f"✅ Data encrypted: {len(encrypted)} bytes")
            
            decrypted = security.decrypt_sensitive_data(encrypted)
            print(f"✅ Data decrypted: {decrypted}")
            
            print("🎯 Security framework operational!")
            
        else:
            print("❌ Authentication failed")
            
    except Exception as e:
        print(f"❌ Security demo failed: {e}")
