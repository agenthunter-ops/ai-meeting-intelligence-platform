# AI Meeting Intelligence Platform - Part 8: Security & Compliance

## Table of Contents
1. [Security Architecture Overview](#security-architecture-overview)
2. [Authentication & Authorization](#authentication--authorization)
3. [Data Protection & Privacy](#data-protection--privacy)
4. [API Security](#api-security)
5. [Infrastructure Security](#infrastructure-security)
6. [Compliance Frameworks](#compliance-frameworks)
7. [Security Monitoring](#security-monitoring)
8. [Incident Response](#incident-response)

---

## Security Architecture Overview

### Zero Trust Security Model

Our security architecture follows a Zero Trust approach, assuming no implicit trust within the system perimeter. Every request is authenticated, authorized, and verified regardless of location or origin.

```
┌─────────────────────────────────────────────────────────────────┐
│                    ZERO TRUST SECURITY LAYERS                   │
├─────────────────┬─────────────────┬─────────────────┬───────────┤
│   IDENTITY      │   DEVICE        │   NETWORK       │   DATA    │
│   VERIFICATION  │   SECURITY      │   MICRO-        │   ENCRYPT │
│                 │                 │   SEGMENTATION  │   TION    │
├─────────────────┼─────────────────┼─────────────────┼───────────┤
│ • Multi-Factor  │ • Device        │ • Service Mesh  │ • E2E     │
│   Authentication│   Certificates  │ • Firewalls     │   Encrypt │
│ • RBAC          │ • EDR/XDR       │ • VPNs          │ • At Rest │
│ • SAML/OIDC     │ • Compliance    │ • Zero Trust    │ • In       │
│ • Session Mgmt  │   Monitoring    │   Networks      │   Transit │
└─────────────────┴─────────────────┴─────────────────┴───────────┘
```

### Security Implementation Framework

```python
# backend/security/framework.py - Security framework implementation

import hashlib
import secrets
import hmac
import jwt
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.serialization import Encoding, PrivateFormat, NoEncryption
import base64
import os
import logging
from enum import Enum

logger = logging.getLogger(__name__)

class SecurityLevel(Enum):
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"

class Permission(Enum):
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    ADMIN = "admin"
    UPLOAD = "upload"
    PROCESS = "process"

class CryptographyManager:
    """Handles all cryptographic operations."""
    
    def __init__(self):
        self.master_key = self._get_or_create_master_key()
        self.fernet = Fernet(self.master_key)
        self.rsa_key_pair = self._get_or_create_rsa_keys()
        
    def _get_or_create_master_key(self) -> bytes:
        """Get or create master encryption key."""
        
        key_file = "security/master.key"
        
        if os.path.exists(key_file):
            with open(key_file, 'rb') as f:
                return f.read()
        else:
            # Generate new key
            key = Fernet.generate_key()
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(key_file), exist_ok=True)
            
            # Save key securely
            with open(key_file, 'wb') as f:
                f.write(key)
            
            # Set restrictive permissions
            os.chmod(key_file, 0o600)
            
            logger.info("Generated new master encryption key")
            return key
    
    def _get_or_create_rsa_keys(self) -> Dict[str, Any]:
        """Get or create RSA key pair for asymmetric encryption."""
        
        private_key_file = "security/private_key.pem"
        public_key_file = "security/public_key.pem"
        
        if os.path.exists(private_key_file) and os.path.exists(public_key_file):
            from cryptography.hazmat.primitives.serialization import load_pem_private_key, load_pem_public_key
            
            with open(private_key_file, 'rb') as f:
                private_key = load_pem_private_key(f.read(), password=None)
            
            with open(public_key_file, 'rb') as f:
                public_key = load_pem_public_key(f.read())
            
            return {"private": private_key, "public": public_key}
        
        else:
            # Generate new RSA key pair
            private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=4096
            )
            public_key = private_key.public_key()
            
            # Serialize keys
            private_pem = private_key.private_bytes(
                encoding=Encoding.PEM,
                format=PrivateFormat.PKCS8,
                encryption_algorithm=NoEncryption()
            )
            
            public_pem = public_key.public_bytes(
                encoding=Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            )
            
            # Save keys
            os.makedirs("security", exist_ok=True)
            
            with open(private_key_file, 'wb') as f:
                f.write(private_pem)
            os.chmod(private_key_file, 0o600)
            
            with open(public_key_file, 'wb') as f:
                f.write(public_pem)
            os.chmod(public_key_file, 0o644)
            
            logger.info("Generated new RSA key pair")
            return {"private": private_key, "public": public_key}
    
    def encrypt_symmetric(self, data: Union[str, bytes]) -> str:
        """Encrypt data using symmetric encryption."""
        
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        encrypted_data = self.fernet.encrypt(data)
        return base64.b64encode(encrypted_data).decode('utf-8')
    
    def decrypt_symmetric(self, encrypted_data: str) -> str:
        """Decrypt data using symmetric encryption."""
        
        try:
            encrypted_bytes = base64.b64decode(encrypted_data.encode('utf-8'))
            decrypted_data = self.fernet.decrypt(encrypted_bytes)
            return decrypted_data.decode('utf-8')
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            raise ValueError("Invalid encrypted data")
    
    def encrypt_asymmetric(self, data: Union[str, bytes]) -> str:
        """Encrypt data using asymmetric encryption (RSA)."""
        
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        encrypted_data = self.rsa_key_pair["public"].encrypt(
            data,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        
        return base64.b64encode(encrypted_data).decode('utf-8')
    
    def decrypt_asymmetric(self, encrypted_data: str) -> str:
        """Decrypt data using asymmetric encryption (RSA)."""
        
        try:
            encrypted_bytes = base64.b64decode(encrypted_data.encode('utf-8'))
            decrypted_data = self.rsa_key_pair["private"].decrypt(
                encrypted_bytes,
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )
            return decrypted_data.decode('utf-8')
        except Exception as e:
            logger.error(f"Asymmetric decryption failed: {e}")
            raise ValueError("Invalid encrypted data")
    
    def hash_password(self, password: str, salt: Optional[bytes] = None) -> Dict[str, str]:
        """Hash password using PBKDF2."""
        
        if salt is None:
            salt = secrets.token_bytes(32)
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,  # OWASP recommended minimum
        )
        
        password_hash = kdf.derive(password.encode('utf-8'))
        
        return {
            "hash": base64.b64encode(password_hash).decode('utf-8'),
            "salt": base64.b64encode(salt).decode('utf-8')
        }
    
    def verify_password(self, password: str, stored_hash: str, stored_salt: str) -> bool:
        """Verify password against stored hash."""
        
        try:
            salt = base64.b64decode(stored_salt.encode('utf-8'))
            stored_password_hash = base64.b64decode(stored_hash.encode('utf-8'))
            
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
            )
            
            # This will raise an exception if passwords don't match
            kdf.verify(password.encode('utf-8'), stored_password_hash)
            return True
            
        except Exception:
            return False
    
    def generate_secure_token(self, length: int = 32) -> str:
        """Generate cryptographically secure random token."""
        return secrets.token_urlsafe(length)
    
    def create_hmac_signature(self, data: str, secret_key: str) -> str:
        """Create HMAC signature for data integrity."""
        
        signature = hmac.new(
            secret_key.encode('utf-8'),
            data.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        return signature
    
    def verify_hmac_signature(self, data: str, signature: str, secret_key: str) -> bool:
        """Verify HMAC signature."""
        
        expected_signature = self.create_hmac_signature(data, secret_key)
        return hmac.compare_digest(signature, expected_signature)

class DataClassification:
    """Handles data classification and protection levels."""
    
    @staticmethod
    def classify_meeting_data(meeting_data: Dict[str, Any]) -> SecurityLevel:
        """Classify meeting data based on content and metadata."""
        
        # Check for sensitive keywords
        sensitive_keywords = [
            "confidential", "secret", "private", "internal",
            "salary", "compensation", "legal", "contract",
            "merger", "acquisition", "layoff", "termination"
        ]
        
        text_content = ""
        if "title" in meeting_data:
            text_content += meeting_data["title"].lower()
        
        if "transcript" in meeting_data:
            text_content += " " + meeting_data["transcript"].lower()
        
        # Check for sensitive content
        for keyword in sensitive_keywords:
            if keyword in text_content:
                return SecurityLevel.CONFIDENTIAL
        
        # Check meeting type
        sensitive_meeting_types = ["executive", "board", "legal", "hr"]
        if meeting_data.get("meeting_type", "").lower() in sensitive_meeting_types:
            return SecurityLevel.RESTRICTED
        
        # Default classification
        return SecurityLevel.INTERNAL
    
    @staticmethod
    def get_encryption_requirements(security_level: SecurityLevel) -> Dict[str, Any]:
        """Get encryption requirements based on security level."""
        
        requirements = {
            SecurityLevel.PUBLIC: {
                "encrypt_at_rest": False,
                "encrypt_in_transit": True,
                "key_rotation_days": 365
            },
            SecurityLevel.INTERNAL: {
                "encrypt_at_rest": True,
                "encrypt_in_transit": True,
                "key_rotation_days": 180
            },
            SecurityLevel.CONFIDENTIAL: {
                "encrypt_at_rest": True,
                "encrypt_in_transit": True,
                "key_rotation_days": 90,
                "additional_auth_required": True
            },
            SecurityLevel.RESTRICTED: {
                "encrypt_at_rest": True,
                "encrypt_in_transit": True,
                "key_rotation_days": 30,
                "additional_auth_required": True,
                "audit_all_access": True
            }
        }
        
        return requirements.get(security_level, requirements[SecurityLevel.INTERNAL])

# Global security instances
crypto_manager = CryptographyManager()
data_classifier = DataClassification()
```

---

## Authentication & Authorization

### JWT-Based Authentication System

```python
# backend/security/auth.py - Authentication and authorization system

import jwt
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
from fastapi import HTTPException, status, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from passlib.context import CryptContext
import redis
import json
import logging
from enum import Enum
from dataclasses import dataclass

logger = logging.getLogger(__name__)

class UserRole(Enum):
    ADMIN = "admin"
    USER = "user"
    VIEWER = "viewer"
    SERVICE = "service"

@dataclass
class User:
    id: str
    username: str
    email: str
    roles: List[UserRole]
    permissions: List[Permission]
    is_active: bool = True
    last_login: Optional[datetime] = None
    mfa_enabled: bool = False

class AuthenticationManager:
    """Handles user authentication and JWT token management."""
    
    def __init__(self, secret_key: str, redis_client=None):
        self.secret_key = secret_key
        self.algorithm = "HS256"
        self.access_token_expire_minutes = 30
        self.refresh_token_expire_days = 7
        self.redis_client = redis_client
        
        # Password hashing
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        
        # Rate limiting
        self.login_attempts = {}  # In production, use Redis
        self.max_login_attempts = 5
        self.lockout_duration = timedelta(minutes=15)
    
    def create_access_token(self, user: User, expires_delta: Optional[timedelta] = None) -> str:
        """Create JWT access token."""
        
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=self.access_token_expire_minutes)
        
        to_encode = {
            "sub": user.id,
            "username": user.username,
            "email": user.email,
            "roles": [role.value for role in user.roles],
            "permissions": [perm.value for perm in user.permissions],
            "exp": expire,
            "iat": datetime.utcnow(),
            "type": "access"
        }
        
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        
        # Store token in Redis for revocation capability
        if self.redis_client:
            self.redis_client.setex(
                f"access_token:{user.id}:{encoded_jwt[:20]}",
                int(timedelta(minutes=self.access_token_expire_minutes).total_seconds()),
                json.dumps(to_encode)
            )
        
        return encoded_jwt
    
    def create_refresh_token(self, user: User) -> str:
        """Create JWT refresh token."""
        
        expire = datetime.utcnow() + timedelta(days=self.refresh_token_expire_days)
        
        to_encode = {
            "sub": user.id,
            "exp": expire,
            "iat": datetime.utcnow(),
            "type": "refresh"
        }
        
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        
        # Store refresh token
        if self.redis_client:
            self.redis_client.setex(
                f"refresh_token:{user.id}",
                int(timedelta(days=self.refresh_token_expire_days).total_seconds()),
                encoded_jwt
            )
        
        return encoded_jwt
    
    def verify_token(self, token: str, token_type: str = "access") -> Dict[str, Any]:
        """Verify and decode JWT token."""
        
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            
            # Verify token type
            if payload.get("type") != token_type:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid token type"
                )
            
            # Check if token is revoked (Redis lookup)
            if self.redis_client and token_type == "access":
                user_id = payload.get("sub")
                token_key = f"access_token:{user_id}:{token[:20]}"
                
                if not self.redis_client.exists(token_key):
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="Token revoked"
                    )
            
            return payload
            
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token expired"
            )
        except jwt.JWTError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
    
    def revoke_token(self, token: str, user_id: str):
        """Revoke access token."""
        
        if self.redis_client:
            token_key = f"access_token:{user_id}:{token[:20]}"
            self.redis_client.delete(token_key)
    
    def revoke_all_user_tokens(self, user_id: str):
        """Revoke all tokens for a user."""
        
        if self.redis_client:
            # Remove all access tokens
            access_pattern = f"access_token:{user_id}:*"
            for key in self.redis_client.scan_iter(match=access_pattern):
                self.redis_client.delete(key)
            
            # Remove refresh token
            self.redis_client.delete(f"refresh_token:{user_id}")
    
    def check_rate_limit(self, identifier: str) -> bool:
        """Check if login attempts are within rate limit."""
        
        current_time = datetime.utcnow()
        
        if identifier in self.login_attempts:
            attempts, last_attempt = self.login_attempts[identifier]
            
            # Reset counter if lockout period has passed
            if current_time - last_attempt > self.lockout_duration:
                del self.login_attempts[identifier]
                return True
            
            # Check if max attempts exceeded
            if attempts >= self.max_login_attempts:
                return False
        
        return True
    
    def record_login_attempt(self, identifier: str, success: bool):
        """Record login attempt for rate limiting."""
        
        current_time = datetime.utcnow()
        
        if success:
            # Reset counter on successful login
            if identifier in self.login_attempts:
                del self.login_attempts[identifier]
        else:
            # Increment failed attempts
            if identifier in self.login_attempts:
                attempts, _ = self.login_attempts[identifier]
                self.login_attempts[identifier] = (attempts + 1, current_time)
            else:
                self.login_attempts[identifier] = (1, current_time)

class AuthorizationManager:
    """Handles user authorization and permission checking."""
    
    def __init__(self):
        self.role_permissions = {
            UserRole.ADMIN: [
                Permission.READ, Permission.WRITE, Permission.DELETE,
                Permission.ADMIN, Permission.UPLOAD, Permission.PROCESS
            ],
            UserRole.USER: [
                Permission.READ, Permission.WRITE, Permission.UPLOAD, Permission.PROCESS
            ],
            UserRole.VIEWER: [
                Permission.READ
            ],
            UserRole.SERVICE: [
                Permission.READ, Permission.WRITE, Permission.PROCESS
            ]
        }
    
    def get_user_permissions(self, user: User) -> List[Permission]:
        """Get all permissions for a user based on their roles."""
        
        permissions = set()
        
        # Add role-based permissions
        for role in user.roles:
            permissions.update(self.role_permissions.get(role, []))
        
        # Add explicit user permissions
        permissions.update(user.permissions)
        
        return list(permissions)
    
    def check_permission(self, user: User, required_permission: Permission) -> bool:
        """Check if user has required permission."""
        
        user_permissions = self.get_user_permissions(user)
        return required_permission in user_permissions
    
    def check_resource_access(self, user: User, resource_id: str, permission: Permission) -> bool:
        """Check if user can access specific resource."""
        
        # Check basic permission
        if not self.check_permission(user, permission):
            return False
        
        # Additional resource-specific checks can be added here
        # For example, checking if user owns the resource or has shared access
        
        return True
    
    def require_permission(self, permission: Permission):
        """Decorator to require specific permission."""
        
        def decorator(func):
            def wrapper(*args, **kwargs):
                # Get current user from context
                user = kwargs.get('current_user')
                if not user:
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="Authentication required"
                    )
                
                if not self.check_permission(user, permission):
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="Insufficient permissions"
                    )
                
                return func(*args, **kwargs)
            return wrapper
        return decorator

# FastAPI Security Dependencies
security = HTTPBearer()
auth_manager = AuthenticationManager(
    secret_key=os.getenv("SECRET_KEY", "development-secret"),
    redis_client=redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379/0"))
)
authz_manager = AuthorizationManager()

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> User:
    """Get current authenticated user."""
    
    token = credentials.credentials
    payload = auth_manager.verify_token(token)
    
    # In production, fetch user from database
    # For now, reconstruct from token payload
    user = User(
        id=payload["sub"],
        username=payload["username"],
        email=payload["email"],
        roles=[UserRole(role) for role in payload["roles"]],
        permissions=[Permission(perm) for perm in payload["permissions"]]
    )
    
    return user

async def get_current_active_user(current_user: User = Depends(get_current_user)) -> User:
    """Get current active user."""
    
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user"
        )
    
    return current_user

def require_permission(permission: Permission):
    """Dependency to require specific permission."""
    
    def permission_checker(current_user: User = Depends(get_current_active_user)):
        if not authz_manager.check_permission(current_user, permission):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission '{permission.value}' required"
            )
        return current_user
    
    return permission_checker

def require_role(role: UserRole):
    """Dependency to require specific role."""
    
    def role_checker(current_user: User = Depends(get_current_active_user)):
        if role not in current_user.roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role '{role.value}' required"
            )
        return current_user
    
    return role_checker
```

### Multi-Factor Authentication (MFA)

```python
# backend/security/mfa.py - Multi-Factor Authentication implementation

import pyotp
import qrcode
import io
import base64
from typing import Dict, Optional, Tuple
from datetime import datetime, timedelta
import secrets
import logging

logger = logging.getLogger(__name__)

class MFAManager:
    """Handles Multi-Factor Authentication operations."""
    
    def __init__(self, issuer_name: str = "AI Meeting Intelligence"):
        self.issuer_name = issuer_name
        self.backup_codes_count = 10
        
    def generate_secret(self) -> str:
        """Generate a new TOTP secret for a user."""
        return pyotp.random_base32()
    
    def generate_qr_code(self, user_email: str, secret: str) -> str:
        """Generate QR code for TOTP setup."""
        
        totp_uri = pyotp.totp.TOTP(secret).provisioning_uri(
            name=user_email,
            issuer_name=self.issuer_name
        )
        
        # Generate QR code
        qr = qrcode.QRCode(version=1, box_size=10, border=5)
        qr.add_data(totp_uri)
        qr.make(fit=True)
        
        # Create QR code image
        img = qr.make_image(fill_color="black", back_color="white")
        
        # Convert to base64
        img_buffer = io.BytesIO()
        img.save(img_buffer, format='PNG')
        img_str = base64.b64encode(img_buffer.getvalue()).decode()
        
        return f"data:image/png;base64,{img_str}"
    
    def verify_token(self, secret: str, token: str, window: int = 1) -> bool:
        """Verify TOTP token."""
        
        try:
            totp = pyotp.TOTP(secret)
            return totp.verify(token, valid_window=window)
        except Exception as e:
            logger.error(f"MFA token verification failed: {e}")
            return False
    
    def generate_backup_codes(self) -> List[str]:
        """Generate backup codes for account recovery."""
        
        codes = []
        for _ in range(self.backup_codes_count):
            code = secrets.token_hex(4).upper()  # 8-character hex code
            codes.append(f"{code[:4]}-{code[4:]}")  # Format: XXXX-XXXX
        
        return codes
    
    def verify_backup_code(self, stored_codes: List[str], provided_code: str) -> Tuple[bool, List[str]]:
        """Verify backup code and remove it from the list."""
        
        if provided_code in stored_codes:
            # Remove used code
            updated_codes = [code for code in stored_codes if code != provided_code]
            return True, updated_codes
        
        return False, stored_codes

class WebAuthnManager:
    """Handles WebAuthn (FIDO2) authentication."""
    
    def __init__(self):
        self.rp_id = os.getenv("WEBAUTHN_RP_ID", "localhost")
        self.rp_name = "AI Meeting Intelligence Platform"
        
    def generate_registration_options(self, user_id: str, username: str) -> Dict[str, Any]:
        """Generate WebAuthn registration options."""
        
        from webauthn import generate_registration_options
        
        options = generate_registration_options(
            rp_id=self.rp_id,
            rp_name=self.rp_name,
            user_id=user_id.encode('utf-8'),
            user_name=username,
            user_display_name=username
        )
        
        return options
    
    def verify_registration_response(self, credential: Dict[str, Any], challenge: str) -> bool:
        """Verify WebAuthn registration response."""
        
        from webauthn import verify_registration_response
        
        try:
            verification = verify_registration_response(
                credential=credential,
                expected_challenge=challenge.encode('utf-8'),
                expected_origin=f"https://{self.rp_id}",
                expected_rp_id=self.rp_id
            )
            
            return verification.verified
        except Exception as e:
            logger.error(f"WebAuthn registration verification failed: {e}")
            return False
    
    def generate_authentication_options(self, user_id: str) -> Dict[str, Any]:
        """Generate WebAuthn authentication options."""
        
        from webauthn import generate_authentication_options
        
        # Get user's registered credentials (from database)
        # For now, return empty list
        allow_credentials = []
        
        options = generate_authentication_options(
            rp_id=self.rp_id,
            allow_credentials=allow_credentials
        )
        
        return options
    
    def verify_authentication_response(self, credential: Dict[str, Any], challenge: str) -> bool:
        """Verify WebAuthn authentication response."""
        
        from webauthn import verify_authentication_response
        
        try:
            # Get stored credential for this user (from database)
            stored_credential = None  # Fetch from database
            
            verification = verify_authentication_response(
                credential=credential,
                expected_challenge=challenge.encode('utf-8'),
                expected_origin=f"https://{self.rp_id}",
                expected_rp_id=self.rp_id,
                credential_public_key=stored_credential["public_key"],
                credential_current_sign_count=stored_credential["sign_count"]
            )
            
            return verification.verified
        except Exception as e:
            logger.error(f"WebAuthn authentication verification failed: {e}")
            return False

# Global MFA instances
mfa_manager = MFAManager()
webauthn_manager = WebAuthnManager()
```

---

## Data Protection & Privacy

### Privacy-by-Design Implementation

```python
# backend/security/privacy.py - Privacy and data protection

import re
import hashlib
from typing import Dict, List, Any, Optional, Set
from datetime import datetime, timedelta
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class PIIType(Enum):
    EMAIL = "email"
    PHONE = "phone"
    SSN = "ssn"
    CREDIT_CARD = "credit_card"
    NAME = "name"
    ADDRESS = "address"
    IP_ADDRESS = "ip_address"

class DataRetentionPolicy:
    """Defines data retention policies."""
    
    POLICIES = {
        "meeting_recordings": timedelta(days=2555),  # 7 years for business records
        "transcripts": timedelta(days=2555),         # 7 years
        "user_logs": timedelta(days=90),             # 3 months
        "security_logs": timedelta(days=365),        # 1 year
        "analytics_data": timedelta(days=730),       # 2 years
        "user_profiles": None,                       # Indefinite (until user deletion)
        "session_data": timedelta(hours=24),         # 24 hours
        "temp_files": timedelta(hours=1),            # 1 hour
    }
    
    @classmethod
    def get_retention_period(cls, data_type: str) -> Optional[timedelta]:
        """Get retention period for data type."""
        return cls.POLICIES.get(data_type)
    
    @classmethod
    def should_purge(cls, data_type: str, created_at: datetime) -> bool:
        """Check if data should be purged based on retention policy."""
        
        retention_period = cls.get_retention_period(data_type)
        
        if retention_period is None:
            return False  # Indefinite retention
        
        expiry_date = created_at + retention_period
        return datetime.utcnow() > expiry_date

class PIIDetector:
    """Detects and handles Personally Identifiable Information."""
    
    def __init__(self):
        self.patterns = {
            PIIType.EMAIL: re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            PIIType.PHONE: re.compile(r'\b(?:\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})\b'),
            PIIType.SSN: re.compile(r'\b\d{3}-?\d{2}-?\d{4}\b'),
            PIIType.CREDIT_CARD: re.compile(r'\b(?:\d{4}[-\s]?){3}\d{4}\b'),
            PIIType.IP_ADDRESS: re.compile(r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'),
        }
        
        # Common name patterns (basic implementation)
        self.name_indicators = [
            r'\bmy name is\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b',
            r'\bi am\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b',
            r'\bthis is\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b',
        ]
    
    def detect_pii(self, text: str) -> Dict[PIIType, List[str]]:
        """Detect PII in text and return found instances."""
        
        detected_pii = {}
        
        # Detect using regex patterns
        for pii_type, pattern in self.patterns.items():
            matches = pattern.findall(text)
            if matches:
                detected_pii[pii_type] = matches
        
        # Detect names using context
        names = self._detect_names(text)
        if names:
            detected_pii[PIIType.NAME] = names
        
        return detected_pii
    
    def _detect_names(self, text: str) -> List[str]:
        """Detect names using contextual patterns."""
        
        names = []
        
        for pattern in self.name_indicators:
            matches = re.findall(pattern, text, re.IGNORECASE)
            names.extend(matches)
        
        return list(set(names))  # Remove duplicates
    
    def anonymize_text(self, text: str, replacement_strategy: str = "mask") -> str:
        """Anonymize PII in text."""
        
        anonymized_text = text
        
        for pii_type, pattern in self.patterns.items():
            if replacement_strategy == "mask":
                anonymized_text = pattern.sub(f"[{pii_type.value.upper()}]", anonymized_text)
            elif replacement_strategy == "hash":
                def hash_match(match):
                    return hashlib.sha256(match.group().encode()).hexdigest()[:8]
                anonymized_text = pattern.sub(hash_match, anonymized_text)
            elif replacement_strategy == "remove":
                anonymized_text = pattern.sub("", anonymized_text)
        
        # Clean up extra spaces
        anonymized_text = re.sub(r'\s+', ' ', anonymized_text).strip()
        
        return anonymized_text

class DataProcessor:
    """Handles data processing with privacy controls."""
    
    def __init__(self):
        self.pii_detector = PIIDetector()
        self.crypto_manager = crypto_manager
        
    def process_meeting_data(self, meeting_data: Dict[str, Any], user_consent: Dict[str, bool]) -> Dict[str, Any]:
        """Process meeting data with privacy controls."""
        
        processed_data = meeting_data.copy()
        
        # Classify data security level
        security_level = data_classifier.classify_meeting_data(meeting_data)
        
        # Get encryption requirements
        encryption_req = data_classifier.get_encryption_requirements(security_level)
        
        # Process transcript if present
        if "transcript" in processed_data:
            processed_data["transcript"] = self._process_transcript(
                processed_data["transcript"],
                user_consent,
                encryption_req
            )
        
        # Process segments
        if "segments" in processed_data:
            processed_data["segments"] = [
                self._process_segment(segment, user_consent, encryption_req)
                for segment in processed_data["segments"]
            ]
        
        # Encrypt sensitive fields if required
        if encryption_req.get("encrypt_at_rest"):
            processed_data = self._encrypt_sensitive_fields(processed_data)
        
        # Add privacy metadata
        processed_data["privacy_metadata"] = {
            "security_level": security_level.value,
            "pii_detected": self._contains_pii(meeting_data),
            "encryption_applied": encryption_req.get("encrypt_at_rest", False),
            "processed_at": datetime.utcnow().isoformat(),
            "consent_version": "1.0"
        }
        
        return processed_data
    
    def _process_transcript(self, transcript: str, consent: Dict[str, bool], encryption_req: Dict) -> str:
        """Process transcript text with privacy controls."""
        
        # Detect PII
        detected_pii = self.pii_detector.detect_pii(transcript)
        
        if detected_pii and not consent.get("pii_processing", False):
            # User hasn't consented to PII processing, anonymize
            return self.pii_detector.anonymize_text(transcript, "mask")
        
        return transcript
    
    def _process_segment(self, segment: Dict[str, Any], consent: Dict[str, bool], encryption_req: Dict) -> Dict[str, Any]:
        """Process individual segment with privacy controls."""
        
        processed_segment = segment.copy()
        
        # Process segment text
        if "text" in processed_segment:
            processed_segment["text"] = self._process_transcript(
                processed_segment["text"],
                consent,
                encryption_req
            )
        
        # Anonymize speaker information if not consented
        if not consent.get("speaker_identification", False):
            if "speaker_name" in processed_segment:
                processed_segment["speaker_name"] = f"Speaker_{segment.get('speaker_id', 'Unknown')}"
        
        return processed_segment
    
    def _encrypt_sensitive_fields(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Encrypt sensitive fields in the data."""
        
        sensitive_fields = ["transcript", "summary", "key_points"]
        encrypted_data = data.copy()
        
        for field in sensitive_fields:
            if field in encrypted_data and encrypted_data[field]:
                if isinstance(encrypted_data[field], str):
                    encrypted_data[field] = self.crypto_manager.encrypt_symmetric(encrypted_data[field])
                elif isinstance(encrypted_data[field], list):
                    encrypted_data[field] = [
                        self.crypto_manager.encrypt_symmetric(item) if isinstance(item, str) else item
                        for item in encrypted_data[field]
                    ]
        
        return encrypted_data
    
    def _contains_pii(self, data: Dict[str, Any]) -> bool:
        """Check if data contains PII."""
        
        text_content = ""
        
        if "transcript" in data:
            text_content += data["transcript"]
        
        if "segments" in data:
            for segment in data["segments"]:
                if "text" in segment:
                    text_content += " " + segment["text"]
        
        detected_pii = self.pii_detector.detect_pii(text_content)
        return len(detected_pii) > 0

class ConsentManager:
    """Manages user consent for data processing."""
    
    def __init__(self):
        self.consent_version = "1.0"
        self.required_consents = [
            "data_processing",
            "audio_storage",
            "transcript_generation",
            "ai_analysis"
        ]
        
        self.optional_consents = [
            "pii_processing",
            "speaker_identification",
            "sentiment_analysis",
            "analytics",
            "service_improvement"
        ]
    
    def get_consent_form(self) -> Dict[str, Any]:
        """Get consent form structure."""
        
        return {
            "version": self.consent_version,
            "required_consents": [
                {
                    "key": "data_processing",
                    "title": "Data Processing",
                    "description": "Process your meeting data to provide transcription and analysis services.",
                    "required": True
                },
                {
                    "key": "audio_storage",
                    "title": "Audio Storage",
                    "description": "Store audio files temporarily for processing purposes.",
                    "required": True
                },
                {
                    "key": "transcript_generation",
                    "title": "Transcript Generation",
                    "description": "Generate text transcripts from your audio recordings.",
                    "required": True
                },
                {
                    "key": "ai_analysis",
                    "title": "AI Analysis",
                    "description": "Use AI to analyze your meetings for insights and summaries.",
                    "required": True
                }
            ],
            "optional_consents": [
                {
                    "key": "pii_processing",
                    "title": "Personal Information Processing",
                    "description": "Process personal information (names, emails, etc.) mentioned in meetings.",
                    "required": False
                },
                {
                    "key": "speaker_identification",
                    "title": "Speaker Identification",
                    "description": "Identify and track individual speakers throughout the meeting.",
                    "required": False
                },
                {
                    "key": "sentiment_analysis",
                    "title": "Sentiment Analysis",
                    "description": "Analyze emotional tone and sentiment in meeting conversations.",
                    "required": False
                },
                {
                    "key": "analytics",
                    "title": "Usage Analytics",
                    "description": "Collect anonymized usage data to improve our services.",
                    "required": False
                },
                {
                    "key": "service_improvement",
                    "title": "Service Improvement",
                    "description": "Use your data to improve our AI models and services.",
                    "required": False
                }
            ]
        }
    
    def validate_consent(self, consent_data: Dict[str, bool]) -> Tuple[bool, List[str]]:
        """Validate user consent data."""
        
        errors = []
        
        # Check required consents
        for consent_key in self.required_consents:
            if not consent_data.get(consent_key, False):
                errors.append(f"Required consent '{consent_key}' not provided")
        
        # Check if all consent keys are valid
        valid_keys = set(self.required_consents + self.optional_consents)
        for key in consent_data.keys():
            if key not in valid_keys:
                errors.append(f"Unknown consent key: {key}")
        
        return len(errors) == 0, errors
    
    def record_consent(self, user_id: str, consent_data: Dict[str, bool]) -> Dict[str, Any]:
        """Record user consent with timestamp and version."""
        
        is_valid, errors = self.validate_consent(consent_data)
        
        if not is_valid:
            raise ValueError(f"Invalid consent data: {errors}")
        
        consent_record = {
            "user_id": user_id,
            "consent_version": self.consent_version,
            "consents": consent_data,
            "timestamp": datetime.utcnow().isoformat(),
            "ip_address": None,  # Should be captured from request
            "user_agent": None   # Should be captured from request
        }
        
        # Store consent record in database
        # Implementation depends on your database setup
        
        return consent_record

# Global privacy instances
data_processor = DataProcessor()
consent_manager = ConsentManager()
```

This completes the first major section of Part 8, covering security architecture, authentication & authorization, and data protection & privacy. The implementation demonstrates:

1. **Zero Trust Security Model** with comprehensive authentication
2. **JWT-based authentication** with refresh tokens and revocation
3. **Multi-Factor Authentication** including TOTP and WebAuthn
4. **Role-based access control** with granular permissions
5. **Privacy-by-design** with PII detection and data anonymization
6. **Consent management** for GDPR/CCPA compliance
7. **Data encryption** for sensitive information protection

Would you like me to continue with the remaining sections covering API Security, Infrastructure Security, Compliance Frameworks, Security Monitoring, and Incident Response?