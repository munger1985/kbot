# tests/test_service_auth.py
import pytest
import json
import time
from datetime import datetime, timezone
from fastapi.testclient import TestClient
from jose import jwt

from main import app  # 假设主应用在main.py
from core.auth.service_registry import ServiceRegistry, AuthMethod
from core.auth.config import SERVICE_SECRET_KEY, ALGORITHM

client = TestClient(app)

# 测试配置
TEST_ADMIN_KEY = "test-admin-secret"
TEST_SERVICE_NAME = "test-service"
TEST_SERVICE_SECRET = None  # 将在测试中设置

@pytest.fixture(autouse=True)
def setup_test():
    """测试前设置环境变量"""
    import os
    os.environ["KBOT_ADMIN_SECRET"] = TEST_ADMIN_KEY
    # 确保每次测试使用干净的注册表
    global service_registry
    service_registry = ServiceRegistry("test_services.json")
    yield
    # 清理测试文件
    import os
    if os.path.exists("test_services.json"):
        os.remove("test_services.json")

class TestServiceRegistration:
    def test_register_service_success(self):
        """测试成功注册服务"""
        response = client.post(
            "/service-auth/register",
            json={
                "service_name": TEST_SERVICE_NAME,
                "auth_method": "preshared",
                "permissions": ["read:data", "write:logs"],
                "metadata": {"environment": "test"}
            },
            headers={"X-Admin-Key": TEST_ADMIN_KEY}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["service_name"] == TEST_SERVICE_NAME
        assert data["auth_method"] == "preshared"
        assert "secret" in data
        global TEST_SERVICE_SECRET
        TEST_SERVICE_SECRET = data["secret"]
    
    def test_register_service_duplicate(self):
        """测试重复注册服务"""
        # 第一次注册
        client.post(
            "/service-auth/register",
            json={"service_name": "duplicate-service"},
            headers={"X-Admin-Key": TEST_ADMIN_KEY}
        )
        
        # 第二次注册相同名称
        response = client.post(
            "/service-auth/register",
            json={"service_name": "duplicate-service"},
            headers={"X-Admin-Key": TEST_ADMIN_KEY}
        )
        
        assert response.status_code == 400
    
    def test_register_service_unauthorized(self):
        """测试无权限注册服务"""
        response = client.post(
            "/service-auth/register",
            json={"service_name": "unauthorized-service"}
        )
        
        assert response.status_code == 401

class TestServiceAuthentication:
    def setup_method(self):
        """每个测试方法前设置"""
        # 注册测试服务
        response = client.post(
            "/service-auth/register",
            json={
                "service_name": TEST_SERVICE_NAME,
                "auth_method": "preshared",
                "permissions": ["read:data", "write:logs"]
            },
            headers={"X-Admin-Key": TEST_ADMIN_KEY}
        )
        global TEST_SERVICE_SECRET
        TEST_SERVICE_SECRET = response.json()["secret"]
    
    def test_get_token_success(self):
        """测试成功获取Token"""
        response = client.post(
            "/service-auth/token",
            json={
                "service_name": TEST_SERVICE_NAME,
                "auth_method": "preshared",
                "credentials": {"secret": TEST_SERVICE_SECRET},
                "requested_permissions": ["read:data"]
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert data["service_name"] == TEST_SERVICE_NAME
        assert data["token_type"] == "bearer"
        assert "read:data" in data["permissions"]
        assert "write:logs" not in data["permissions"]  # 未请求的权限不应包含
    
    def test_get_token_invalid_credentials(self):
        """测试使用无效凭证获取Token"""
        response = client.post(
            "/service-auth/token",
            json={
                "service_name": TEST_SERVICE_NAME,
                "auth_method": "preshared",
                "credentials": {"secret": "wrong-secret"}
            }
        )
        
        assert response.status_code == 401
    
    def test_get_token_nonexistent_service(self):
        """测试不存在的服务获取Token"""
        response = client.post(
            "/service-auth/token",
            json={
                "service_name": "non-existent-service",
                "auth_method": "preshared",
                "credentials": {"secret": "any-secret"}
            }
        )
        
        assert response.status_code == 401

class TestTokenVerification:
    def setup_method(self):
        """每个测试方法前设置"""
        # 注册服务并获取Token
        response = client.post(
            "/service-auth/register",
            json={
                "service_name": TEST_SERVICE_NAME,
                "auth_method": "preshared",
                "permissions": ["read:data", "write:logs"]
            },
            headers={"X-Admin-Key": TEST_ADMIN_KEY}
        )
        secret = response.json()["secret"]
        
        # 获取Token
        response = client.post(
            "/service-auth/token",
            json={
                "service_name": TEST_SERVICE_NAME,
                "auth_method": "preshared",
                "credentials": {"secret": secret}
            }
        )
        self.valid_token = response.json()["access_token"]
    
    def test_verify_token_success(self):
        """测试成功验证Token"""
        response = client.post(
            "/service-auth/verify",
            headers={"Authorization": f"Bearer {self.valid_token}"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["valid"] == True
        assert data["service_name"] == TEST_SERVICE_NAME
    
    def test_verify_token_with_required_permissions(self):
        """测试验证Token并检查权限"""
        response = client.post(
            "/service-auth/verify",
            headers={"Authorization": f"Bearer {self.valid_token}"},
            params={"required_permissions": ["read:data"]}
        )
        
        assert response.status_code == 200
        
        # 测试权限不足
        response = client.post(
            "/service-auth/verify",
            headers={"Authorization": f"Bearer {self.valid_token}"},
            params={"required_permissions": ["admin:all"]}
        )
        
        assert response.status_code == 403
    
    def test_verify_invalid_token(self):
        """测试验证无效Token"""
        response = client.post(
            "/service-auth/verify",
            headers={"Authorization": "Bearer invalid-token"}
        )
        
        assert response.status_code == 401
    
    def test_verify_expired_token(self):
        """测试验证过期Token"""
        # 创建过期的Token
        expired_token = jwt.encode(
            {
                "sub": TEST_SERVICE_NAME,
                "token_type": "service",
                "service_name": TEST_SERVICE_NAME,
                "iss": "kbot_auth_service",
                "aud": ["internal_services"],
                "iat": int(time.time()) - 1000,
                "exp": int(time.time()) - 600,  # 10分钟前过期
                "permissions": ["read:data"]
            },
            SERVICE_SECRET_KEY,
            algorithm=ALGORITHM
        )
        
        response = client.post(
            "/service-auth/verify",
            headers={"Authorization": f"Bearer {expired_token}"}
        )
        
        assert response.status_code == 401

class TestServiceManagement:
    def setup_method(self):
        """每个测试方法前设置"""
        # 注册几个测试服务
        services = ["service-a", "service-b", "service-c"]
        for service in services:
            client.post(
                "/service-auth/register",
                json={"service_name": service},
                headers={"X-Admin-Key": TEST_ADMIN_KEY}
            )
    
    def test_list_services(self):
        """测试列出所有服务"""
        response = client.get(
            "/service-auth/services",
            headers={"X-Admin-Key": TEST_ADMIN_KEY}
        )
        
        assert response.status_code == 200
        services = response.json()
        assert len(services) >= 3
        
        service_names = [s["name"] for s in services]
        assert "service-a" in service_names
        assert "service-b" in service_names
        assert "service-c" in service_names
    
    def test_update_service(self):
        """测试更新服务信息"""
        # 更新服务权限
        response = client.put(
            f"/service-auth/services/service-a",
            params={
                "permissions": ["new:permission"],
                "status": "suspended"
            },
            headers={"X-Admin-Key": TEST_ADMIN_KEY}
        )
        
        assert response.status_code == 200
        
        # 验证更新
        response = client.get(
            "/service-auth/services",
            headers={"X-Admin-Key": TEST_ADMIN_KEY}
        )
        
        services = response.json()
        service_a = next(s for s in services if s["name"] == "service-a")
        assert service_a["permissions"] == ["new:permission"]
        assert service_a["status"] == "suspended"

class TestIntegration:
    def test_full_flow(self):
        """测试完整流程：注册->获取Token->使用Token访问受保护资源"""
        # 1. 注册服务
        register_response = client.post(
            "/service-auth/register",
            json={
                "service_name": "integration-service",
                "auth_method": "preshared",
                "permissions": ["api:read"]
            },
            headers={"X-Admin-Key": TEST_ADMIN_KEY}
        )
        secret = register_response.json()["secret"]
        
        # 2. 获取Token
        token_response = client.post(
            "/service-auth/token",
            json={
                "service_name": "integration-service",
                "auth_method": "preshared",
                "credentials": {"secret": secret},
                "requested_permissions": ["api:read"]
            }
        )
        token = token_response.json()["access_token"]
        
        # 3. 验证Token
        verify_response = client.post(
            "/service-auth/verify",
            headers={"Authorization": f"Bearer {token}"},
            params={"required_permissions": ["api:read"]}
        )
        
        assert verify_response.status_code == 200
        assert verify_response.json()["valid"] == True
        
        # 4. 尝试使用Token访问需要权限的接口
        # 这里假设有一个受保护的API端点
        # protected_response = client.get(
        #     "/api/protected",
        #     headers={"Authorization": f"Bearer {token}"}
        # )
        # assert protected_response.status_code == 200

def test_custom_token_expiry():
    """测试自定义Token过期时间"""
    # 注册服务
    response = client.post(
        "/service-auth/register",
        json={"service_name": "expiry-test-service"},
        headers={"X-Admin-Key": TEST_ADMIN_KEY}
    )
    secret = response.json()["secret"]
    
    # 请求60秒过期的Token
    response = client.post(
        "/service-auth/token",
        json={
            "service_name": "expiry-test-service",
            "auth_method": "preshared",
            "credentials": {"secret": secret},
            "expires_in": 60  # 60秒
        }
    )
    
    assert response.status_code == 200
    token = response.json()["access_token"]
    
    # 解码Token检查过期时间
    payload = jwt.decode(token, SERVICE_SECRET_KEY, algorithms=[ALGORITHM])
    iat = payload["iat"]
    exp = payload["exp"]
    
    # 过期时间应该在60秒左右
    assert abs((exp - iat) - 60) < 5

def test_service_registry_persistence():
    """测试服务注册表的持久化"""
    # 创建新的注册表实例
    registry1 = ServiceRegistry("test_persistence.json")
    registry1.register_service("persistent-service", AuthMethod.PRESHARED)
    
    # 创建另一个实例，应该能读取到之前的数据
    registry2 = ServiceRegistry("test_persistence.json")
    service_info = registry2.get_service("persistent-service")
    
    assert service_info is not None
    assert service_info.name == "persistent-service"
    
    # 清理
    import os
    if os.path.exists("test_persistence.json"):
        os.remove("test_persistence.json")

if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-v"])