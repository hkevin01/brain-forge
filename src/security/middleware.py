"""
Brain-Forge Security Middleware

FastAPI middleware for authentication, authorization, and security.
"""

import time
from typing import Any, Dict

# FastAPI imports
try:
    from fastapi import Depends, HTTPException, Request, Response, status
    from fastapi.middleware.base import BaseHTTPMiddleware
    from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    Request = None
    Response = None
    HTTPException = None
    BaseHTTPMiddleware = None

from core.logger import get_logger
from security.config import ENDPOINT_SECURITY, SecurityConfig
from security.framework import (
    AuthenticationError,
    AuthorizationError,
    PermissionLevel,
    SecurityContext,
    SecurityManager,
)


class SecurityMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware for security enforcement"""
    
    def __init__(self, app, security_manager: SecurityManager):
        """Initialize security middleware"""
        super().__init__(app)
        self.security_manager = security_manager
        self.logger = get_logger(f"{__name__}.SecurityMiddleware")
        
        # Rate limiting storage (in production, use Redis)
        self.rate_limit_storage: Dict[str, Dict[str, Any]] = {}
        
        self.logger.info("Security middleware initialized")
    
    async def dispatch(self, request: Request, call_next):
        """Process request through security middleware"""
        start_time = time.time()
        
        try:
            # Add security headers
            self._add_security_headers(request)
            
            # Check rate limiting
            if not self._check_rate_limit(request):
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Rate limit exceeded"
                )
            
            # Skip authentication for public endpoints
            if self._is_public_endpoint(request.url.path):
                response = await call_next(request)
                return self._add_response_headers(response)
            
            # Authenticate request
            security_context = await self._authenticate_request(request)
            request.state.security_context = security_context
            
            # Authorize request
            if not self._authorize_request(request, security_context):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Insufficient permissions"
                )
            
            # Process request
            response = await call_next(request)
            # Log successful request
            self._log_request(
                request, security_context, time.time() - start_time, True
            )
            
            return self._add_response_headers(response)
            
        except HTTPException:
            # Re-raise HTTP exceptions
            raise
        except Exception as e:
            self.logger.error(f"Security middleware error: {e}")
            
            # Log failed request
            if hasattr(request.state, 'security_context'):
                self._log_request(
                    request, request.state.security_context,
                    time.time() - start_time, False
                )
            
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Internal security error"
            )
    
    def _add_security_headers(self, request: Request):
        """Add security headers to request state"""
        # Headers will be added to response
        pass
    
    def _add_response_headers(self, response: Response) -> Response:
        """Add security headers to response"""
        for header, value in SecurityConfig.SECURITY_HEADERS.items():
            response.headers[header] = value
        return response
    
    def _is_public_endpoint(self, path: str) -> bool:
        """Check if endpoint is public (no authentication required)"""
        public_endpoints = [
            '/docs', '/redoc', '/openapi.json',
            '/health', '/api/v1/health',
            '/api/v1/auth/login'
        ]
        
        return any(path.startswith(endpoint) for endpoint in public_endpoints)
    
    def _check_rate_limit(self, request: Request) -> bool:
        """Check request rate limiting"""
        try:
            client_ip = self._get_client_ip(request)
            path = request.url.path
            current_time = time.time()
            
            # Get rate limit settings for endpoint
            endpoint_config = ENDPOINT_SECURITY.get(path, {})
            rate_limit_type = endpoint_config.get('rate_limit', 'normal')
            
            if rate_limit_type == 'strict':
                limit = SecurityConfig.LOGIN_RATE_LIMIT_PER_MINUTE
            else:
                limit = SecurityConfig.API_RATE_LIMIT_PER_MINUTE
            
            # Initialize client rate limit data
            if client_ip not in self.rate_limit_storage:
                self.rate_limit_storage[client_ip] = {
                    'requests': [],
                    'blocked_until': 0
                }
            
            client_data = self.rate_limit_storage[client_ip]
            
            # Check if client is blocked
            if current_time < client_data['blocked_until']:
                return False
            
            # Clean old requests (older than 1 minute)
            minute_ago = current_time - 60
            client_data['requests'] = [
                req_time for req_time in client_data['requests']
                if req_time > minute_ago
            ]
            
            # Check rate limit
            if len(client_data['requests']) >= limit:
                # Block client for lockout period
                if rate_limit_type == 'strict':
                    config = SecurityConfig
                    lockout_minutes = config.FAILED_LOGIN_LOCKOUT_MINUTES
                    lockout_seconds = lockout_minutes * 60
                    client_data['blocked_until'] = (
                        current_time + lockout_seconds
                    )
                
                self.logger.warning(f"Rate limit exceeded for {client_ip}")
                return False
            
            # Add current request
            client_data['requests'].append(current_time)
            return True
            
        except Exception as e:
            self.logger.error(f"Rate limit check error: {e}")
            return True  # Allow request on error
    
    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address"""
        # Check X-Forwarded-For header (proxy/load balancer)
        forwarded_for = request.headers.get('X-Forwarded-For')
        if forwarded_for:
            return forwarded_for.split(',')[0].strip()
        
        # Check X-Real-IP header
        real_ip = request.headers.get('X-Real-IP')
        if real_ip:
            return real_ip
        
        # Use client host
        return getattr(request.client, 'host', 'unknown')
    
    async def _authenticate_request(self, request: Request) -> SecurityContext:
        """Authenticate request and return security context"""
        try:
            # Extract token from Authorization header
            auth_header = request.headers.get('Authorization')
            if not auth_header:
                raise AuthenticationError("Authorization header missing")
            
            # Parse Bearer token
            if not auth_header.startswith('Bearer '):
                raise AuthenticationError(
                    "Invalid authorization header format"
                )
            
            token = auth_header[7:]  # Remove 'Bearer ' prefix
            
            # Authenticate with security manager
            client_ip = self._get_client_ip(request)
            security_context = self.security_manager.authenticate_request(
                token, client_ip
            )
            
            return security_context
            
        except AuthenticationError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication failed",
                headers={"WWW-Authenticate": "Bearer"}
            )
        except Exception as e:
            self.logger.error(f"Authentication error: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Authentication system error"
            )
    def _authorize_request(self, request, context) -> bool:
        """Check request authorization"""
        try:
            path = request.url.path
            method = request.method
            
            # Get endpoint security requirements
            endpoint_config = ENDPOINT_SECURITY.get(path, {})
            required_permission = endpoint_config.get('permission')
            
            if not required_permission:
                return True  # No specific permission required
            
            # Map permission string to enum
            permission_map = {
                'read': PermissionLevel.READ,
                'write': PermissionLevel.WRITE,
                'delete': PermissionLevel.DELETE,
                'admin': PermissionLevel.ADMIN
            }
            
            permission = permission_map.get(required_permission)
            if not permission:
                self.logger.warning(f"Unknown permission: {required_permission}")
                return False
            
            # Determine resource from path
            resource = self._get_resource_from_path(path)
            
            # Check authorization
            return self.security_manager.authorize_action(
                context, resource, permission
            )
            
        except AuthorizationError:
            return False
        except Exception as e:
            self.logger.error(f"Authorization error: {e}")
            return False
    
    def _get_resource_from_path(self, path: str) -> str:
        """Extract resource name from API path"""
        path_mapping = {
            '/api/v1/brain-data': 'brain_data',
            '/api/v1/acquisition': 'brain_data',
            '/api/v1/processing': 'brain_data',
            '/api/v1/transfer-learning': 'brain_data',
            '/api/v1/users': 'user_management',
            '/api/v1/admin': 'system_config',
            '/api/v1/export': 'data_export'
        }
        
        for api_path, resource in path_mapping.items():
            if path.startswith(api_path):
                return resource
        
        return 'api_access'  # Default resource
    
    def _log_request(self, request: Request, context: SecurityContext,
                    duration: float, success: bool):
        """Log request for audit purposes"""
        try:
            if context and context.user:
                self.security_manager.audit.log_data_access(
                    context.user,
                    request.url.path,
                    f"{request.method}",
                    f"Duration: {duration:.3f}s, Success: {success}"
                )
        except Exception as e:
            self.logger.error(f"Request logging error: {e}")


class SecurityDependency:
    """FastAPI dependency for security context injection"""
    
    def __init__(self, security_manager: SecurityManager):
        """Initialize security dependency"""
        self.security_manager = security_manager
        self.security_scheme = HTTPBearer()
        self.logger = get_logger(f"{__name__}.SecurityDependency")
    
    async def __call__(self, credentials: HTTPAuthorizationCredentials = 
                      Depends(HTTPBearer())) -> SecurityContext:
        """Dependency injection for security context"""
        try:
            # Authenticate token
            security_context = self.security_manager.authenticate_request(
                credentials.credentials
            )
            
            return security_context
            
        except AuthenticationError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials",
                headers={"WWW-Authenticate": "Bearer"}
            )
        except Exception as e:
            self.logger.error(f"Security dependency error: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Authentication system error"
            )


def require_permission(permission: PermissionLevel, resource: str = None):
    """Decorator for endpoint permission requirements"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Get security context from request
            security_context = kwargs.get('security_context')
            if not security_context:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Authentication required"
                )
            
            # Use provided resource or derive from function name
            target_resource = resource or func.__name__.replace('_', '')
            
            # Check permission (will be handled by middleware)
            return func(*args, **kwargs)
        return wrapper
    return decorator


def create_security_middleware(security_manager: SecurityManager):
    """Create security middleware instance"""
    if not FASTAPI_AVAILABLE:
        raise ImportError("FastAPI not available for security middleware")
    
    return SecurityMiddleware(None, security_manager)


def create_security_dependency(security_manager: SecurityManager):
    """Create security dependency instance"""
    if not FASTAPI_AVAILABLE:
        raise ImportError("FastAPI not available for security dependency")
    
    return SecurityDependency(security_manager)
