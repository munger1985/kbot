from pydantic import BaseModel, Field

class AccessorForm(BaseModel):
    """API security accessor form model.
    
    This model defines the data structure for API accessor authentication 
    and management, including application ID, accessor information, 
    credentials and status.
    """
    app_id: int = Field(..., description="Application ID (unique identifier for the app)")
    accessor: str = Field(..., description="Accessor name/identifier")
    accessor_type: int = Field(..., description="Accessor type (e.g., 1 for system, 2 for user)")
    plain_password: str = Field(..., description="Plaintext password for authentication")
    status: int = Field(0, description="Accessor status (0: inactive, 1: active)")
    descs: str|None = Field(None, description="Additional description of the accessor")
    by: str|None = Field(None, description="Creator/operator of this accessor record")

class ChangePasswordForm(BaseModel):
    """Password change form model.
    
    This model defines the data structure for user password modification 
    requests, including identity verification and new password information.
    """
    username: str = Field(..., description="Username (unique user identifier)")
    old_password: str = Field(..., description="Current/old password (for verification)")
    new_password: str = Field(..., description="New password to be set")