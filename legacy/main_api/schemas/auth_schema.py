from pydantic import BaseModel, Field

class UserRegisterRequest(BaseModel):
    """Request model for user registration.
    
    This model defines the data structure required for a user to register an account.
    """
    username: str = Field(..., description="Username (unique user identifier)")
    email: str = Field(..., description="Email address (used for account verification and communication)")
    password: str = Field(..., description="Password (should follow security requirements)")


class LoginResponse(BaseModel):
    """Response model for user login.
    
    This model defines the data structure returned to the client after successful login,
    including authentication tokens and user information.
    """
    access_token: str = Field(..., description="Access token (used for API authentication)")
    refresh_token: str = Field(..., description="Refresh token (used to obtain new access tokens)")
    token_type: str = Field(..., description="Token type (e.g., 'Bearer')")
    user_id: int = Field(..., description="User ID (unique numeric identifier for the user)")
    username: str = Field(..., description="Username (user's display name/identifier)")
    expires_in: int = Field(..., description="Expiration time (seconds until access token expires)")

class ChangePasswordRequest(BaseModel):
    """Request model for password change.
    
    This model defines the data structure required for a user to update their password.
    """
    username: str = Field(..., description="Username (identifier of the user changing password)")
    # old_password: str = Field(..., description="Old password (for identity verification)")
    new_password: str = Field(..., description="New password (must meet password policy requirements)")


class CreateAPIKeyRequest(BaseModel):
    """Request model for creating API key.
    
    This model defines the data structure required to generate a new API key for service access.
    """
    service_id: int = Field(..., description="Service ID (unique identifier of the target service)")
    name: str = Field(..., description="API key name (human-readable label for the API key)")
    scopes: list[str] | None = Field(None, description="Scopes (list of permissions granted to the API key)")
    expires_days: int | None = Field(None, description="Expiration time (number of days until API key expires)")
    allowed_ips: list[str] | None = Field(None, description="Allowed IP list (list of IP addresses permitted to use the API key)")
    rate_limit: int = Field(0, description="Rate limit (maximum requests per second, 0 = unlimited)")
    created_by: str = Field(..., description="Created by (username of the user creating the API key)")

class CreateServiceRequest(BaseModel):
    """Request model for creating service.
    
    This model defines the data structure required to register a new service in the system.
    """
    service_code: str = Field(..., description="Service code (unique string identifier for the service)")
    name: str = Field(..., description="Service name (human-readable name of the service)")
    service_type: str = Field("internal", description="Service type (default: 'internal', e.g., 'external' for public services)")
    description: str | None = Field(None, description="Service description (optional detailed explanation of the service)")
    owner: str | None = Field(None, description="Service owner (username/team responsible for the service)")
    contact_email: str | None = Field(None, description="Contact email (email for service-related inquiries)")