"""
Brain-Forge API Security Integration

Integration of security framework with REST API endpoints.
"""

# FastAPI imports
try:
    from fastapi import Depends, FastAPI, HTTPException, status
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    FastAPI = None
    Depends = None
    HTTPException = None
    status = None
    HTTPBearer = None
    HTTPAuthorizationCredentials = None
    CORSMiddleware = None

from core.logger import get_logger
from security.config import SecurityConfig
from security.framework import PermissionLevel, SecurityContext, SecurityManager
from security.middleware import SecurityDependency, create_security_middleware


class SecureBrainForgeAPI:
    """Secure Brain-Forge API with integrated security"""
    
    def __init__(self):
        """Initialize secure API"""
        self.logger = get_logger(f"{__name__}.SecureBrainForgeAPI")
        
        if not FASTAPI_AVAILABLE:
            raise ImportError("FastAPI not available")
        
        # Initialize security manager
        self.security_manager = SecurityManager()
        
        # Create FastAPI app
        self.app = FastAPI(
            title="Brain-Forge Secure API",
            description="Multi-Modal BCI Platform with Security",
            version="1.0.0",
            docs_url="/docs" if SecurityConfig.ENABLE_API_DOCS else None,
            redoc_url="/redoc" if SecurityConfig.ENABLE_API_DOCS else None
        )
        
        # Configure CORS
        self._configure_cors()
        
        # Add security middleware
        self._add_security_middleware()
        
        # Create security dependency
        self.get_security_context = SecurityDependency(self.security_manager)
        
        # Register routes
        self._register_routes()
        
        self.logger.info("Secure Brain-Forge API initialized")
    
    def _configure_cors(self):
        """Configure CORS middleware"""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=SecurityConfig.CORS_ORIGINS,
            allow_credentials=True,
            allow_methods=["GET", "POST", "PUT", "DELETE"],
            allow_headers=["*"]
        )
    
    def _add_security_middleware(self):
        """Add security middleware to app"""
        security_middleware = create_security_middleware(self.security_manager)
        self.app.middleware("http")(security_middleware.dispatch)
    
    def _register_routes(self):
        """Register API routes with security"""
        
        # Health endpoint (public)
        @self.app.get("/health")
        @self.app.get("/api/v1/health")
        async def health_check():
            """Health check endpoint"""
            return {"status": "healthy", "service": "brain-forge-api"}
        
        # Authentication endpoints
        @self.app.post("/api/v1/auth/login")
        async def login(credentials: dict):
            """User login"""
            try:
                username = credentials.get("username")
                password = credentials.get("password")
                
                if not username or not password:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Username and password required"
                    )
                
                # Authenticate user
                user = self.security_manager.auth.authenticate_user(
                    username, password
                )
                
                if not user:
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="Invalid credentials"
                    )
                
                # Generate token
                token = self.security_manager.auth.generate_token(user)
                
                # Log successful login
                self.security_manager.audit.log_authentication(
                    user, "login", True, "API login"
                )
                
                return {
                    "access_token": token,
                    "token_type": "bearer",
                    "user_id": user.user_id,
                    "permissions": user.permissions
                }
                
            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"Login error: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Authentication system error"
                )
        
        @self.app.post("/api/v1/auth/logout")
        async def logout(
            context: SecurityContext = Depends(self.get_security_context)
        ):
            """User logout"""
            try:
                # Revoke token
                self.security_manager.auth.revoke_token(context.token)
                
                # Log logout
                if context.user:
                    self.security_manager.audit.log_authentication(
                        context.user, "logout", True, "API logout"
                    )
                
                return {"message": "Logged out successfully"}
                
            except Exception as e:
                self.logger.error(f"Logout error: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Logout error"
                )
        
        # Brain data endpoints
        @self.app.get("/api/v1/brain-data")
        async def get_brain_data(
            context: SecurityContext = Depends(self.get_security_context)
        ):
            """Get brain data (requires read permission)"""
            try:
                # Check read permission
                if not self.security_manager.authorize_action(
                    context, "brain_data", PermissionLevel.READ
                ):
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="Insufficient permissions"
                    )
                
                # Log data access
                self.security_manager.audit.log_data_access(
                    context.user, "brain_data", "read", "API access"
                )
                
                # Mock data response
                return {
                    "message": "Brain data retrieved",
                    "user": context.user.username,
                    "data": "encrypted_brain_data_placeholder"
                }
                
            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"Brain data access error: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Data access error"
                )
        
        @self.app.post("/api/v1/brain-data")
        async def create_brain_data(
            data: dict,
            context: SecurityContext = Depends(self.get_security_context)
        ):
            """Create brain data (requires write permission)"""
            try:
                # Check write permission
                if not self.security_manager.authorize_action(
                    context, "brain_data", PermissionLevel.WRITE
                ):
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="Insufficient permissions"
                    )
                
                # Encrypt sensitive data
                encrypted_data = self.security_manager.encryption.encrypt_data(
                    str(data)
                )
                
                # Log data creation
                self.security_manager.audit.log_data_access(
                    context.user, "brain_data", "create",
                    f"Data size: {len(str(data))} bytes"
                )
                
                return {
                    "message": "Brain data created",
                    "user": context.user.username,
                    "encrypted": True,
                    "data_id": "mock_data_id_123"
                }
                
            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"Brain data creation error: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Data creation error"
                )
        
        # Admin endpoints
        @self.app.get("/api/v1/admin/users")
        async def get_users(
            context: SecurityContext = Depends(self.get_security_context)
        ):
            """Get users (requires admin permission)"""
            try:
                # Check admin permission
                if not self.security_manager.authorize_action(
                    context, "user_management", PermissionLevel.ADMIN
                ):
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="Admin access required"
                    )
                
                # Log admin access
                self.security_manager.audit.log_admin_action(
                    context.user, "list_users", "Admin user listing"
                )
                
                return {
                    "message": "User list retrieved",
                    "admin": context.user.username,
                    "users": ["researcher1", "clinician1", "admin1"]
                }
                
            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"Admin users error: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Admin access error"
                )
        
        # Data export endpoint
        @self.app.get("/api/v1/export/{data_type}")
        async def export_data(
            data_type: str,
            context: SecurityContext = Depends(self.get_security_context)
        ):
            """Export data (requires appropriate permissions)"""
            try:
                # Determine required permission based on data type
                if data_type in ["brain_data", "analysis"]:
                    required_permission = PermissionLevel.READ
                    resource = "data_export"
                else:
                    required_permission = PermissionLevel.ADMIN
                    resource = "system_config"
                
                # Check permission
                if not self.security_manager.authorize_action(
                    context, resource, required_permission
                ):
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="Insufficient permissions for data export"
                    )
                
                # Log export
                self.security_manager.audit.log_data_access(
                    context.user, data_type, "export",
                    f"Export type: {data_type}"
                )
                
                return {
                    "message": f"Data export initiated: {data_type}",
                    "user": context.user.username,
                    "export_id": f"export_{data_type}_{context.user.user_id}"
                }
                
            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"Data export error: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Export error"
                )


def create_secure_api() -> SecureBrainForgeAPI:
    """Create secure Brain-Forge API instance"""
    return SecureBrainForgeAPI()


def get_app() -> FastAPI:
    """Get FastAPI app instance"""
    secure_api = create_secure_api()
    return secure_api.app
