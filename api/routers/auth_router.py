from fastapi import APIRouter, Depends, HTTPException, status, Request
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from api.schemas.auth_schema import *
from api.controllers.auth_controller import AuthController
from core.auth import require_user_token, get_current_user, require_api_key


router = APIRouter(prefix="/auth", tags=["Authentication"])
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")


@router.post("/register", response_model=dict)
async def handle_register(request: UserRegisterRequest):
    """
    ### Description
    Creates a new user account in the system.

    ---
    ### Parameters
    - **username**: Unique identifier for the user.
    - **password**: Secure password (will be hashed).
    - **email**: Valid email address for notifications/recovery.
    """
    return await AuthController.register(request)


@router.post("/login", response_model=LoginResponse)
async def handle_login(
    request: Request,
    form_data: OAuth2PasswordRequestForm = Depends()
):
    """
    ### Description
    Authenticates a user and returns access/refresh tokens.

    ---
    ### Notes
    - Uses **OAuth2 Password Grant** flow.
    - The returned `access_token` should be used in the `Authorization: Bearer <token>` header for subsequent requests.

    ### Error Handling
    - Returns **401 Unauthorized** for invalid credentials.
    """
    login_result = await AuthController.login(request, form_data)
    
    if not login_result:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return login_result


@router.post("/refresh", response_model=dict)
async def handle_refresh_token(refresh_token: str):
    """
    ### Description
    Exchanges a valid **Refresh Token** for a new **Access Token**.
    
    This allows users to stay logged in without re-entering credentials once the access token expires.
    """
    return await AuthController.refresh_token(refresh_token)


@router.post("/logout")
async def handle_logout(
    request: Request,
    auth_info: dict = Depends(require_user_token())
):
    """
    ### Description
    Invalidates the current user session and tokens.
    """
    return await AuthController.logout(request, auth_info)

@router.delete("/remove")
async def handle_remove_user(
    username: str,
    auth_info: dict = Depends(require_user_token())
):
    """
    ### Description
    Removes the current user account.
    """
    return await AuthController.remove_user(username, auth_info)

@router.post("/change-password")
async def handle_change_password(
    request_data: ChangePasswordRequest,
    auth_info: dict = Depends(require_user_token())
):
    """
    ### Description
    Changes the password for the current user.
    """
    return await AuthController.change_password(request_data, auth_info)

@router.post("/service-api-keys", response_model=dict)
async def handle_create_service_api_key(
    request_data: CreateAPIKeyRequest,
    auth_info: dict = Depends(require_user_token())
):
    """
    ### Description
    Generates a new API Key for a specific service. 

    ---
    ### Requirements
    - Must be authenticated as a **User**.
    - API keys are typically used for programmatic/machine-to-machine access.
    """
    return await AuthController.create_service_api_key(request_data, auth_info)


@router.get("/service-api-keys/{service_id}", response_model=list[dict])
async def handle_list_service_api_keys(
    service_id: int,
    active_only: bool = True,
    auth_info: dict = Depends(get_current_user())
):
    """
    ### Description
    Retrieves all API keys associated with a specific service ID.

    - **active_only**: If `True`, filtered to return only non-revoked keys.
    """
    return await AuthController.list_service_api_keys(service_id, auth_info, active_only)


@router.delete("/service-api-keys/{key_id}")
async def handle_revoke_service_api_key(
    key_id: str,
    reason: str | None = None,
    auth_info: dict = Depends(get_current_user())
):
    """
    ### Description
    Immediately revokes an API key, making it invalid for further authentication.
    """
    return await AuthController.revoke_service_api_key(key_id, auth_info, reason)


@router.post("/validate-api-key", response_model=dict)
async def handle_validate_api_key(api_key: str):
    """
    ### Description
    Internal check to verify if a provided API key is valid and active.
    """
    return await AuthController.validate_api_key(api_key)

@router.get("/me", response_model=dict)
async def get_current_user_info(auth_info: dict = Depends(get_current_user())):
    """
    ### Description
    Returns information about the currently authenticated identity (User or Service).
    """
    return auth_info


# Service Management Endpoints
@router.post("/services", response_model=dict)
async def handle_create_service(
    service: CreateServiceRequest,
    auth_info: dict = Depends(require_user_token())
):
    """
    ### Description
    Creates a new service entity. Services are used to group API keys and permissions.
    """
    return await AuthController.create_service(
        service.service_code, 
        service.name, 
        auth_info, 
        service.service_type, 
        service.description, 
        service.owner, 
        service.contact_email
    )


@router.get("/services", response_model=list[dict])
async def handle_list_services(auth_info: dict = Depends(get_current_user())):
    """
    ### Description
    Retrieves a list of all services the current user has permission to view.
    """
    return await AuthController.list_services(auth_info)


# 测试端点
@router.get("/test/user-only")
async def test_user_only(auth_info: dict = Depends(require_user_token())):
    """
    ### Test Endpoint
    Verifies that the request is authenticated via **User Token** (JWT).
    """
    return {
        "message": "User only endpoint",
        "user_id": auth_info["user_id"],
        "username": auth_info["username"]
    }


@router.get("/test/api-key-only")
async def test_api_key_only(auth_info: dict = Depends(require_api_key())):
    """
    ### Test Endpoint
    Verifies that the request is authenticated via **API Key**.
    """
    return {
        "message": "API Key only endpoint",
        "service_id": auth_info["service_id"],
        "service_code": auth_info["service_code"]
    }


@router.get("/test/mixed")
async def test_mixed(auth_info: dict = Depends(get_current_user())):
    """
    ### Test Endpoint
    Verifies that the request is authenticated by **either** a User Token or an API Key.
    """
    return {
        "message": "Mixed authentication endpoint",
        "auth_type": auth_info["type"],
        "details": auth_info
    }