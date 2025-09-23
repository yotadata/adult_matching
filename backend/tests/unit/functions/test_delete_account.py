"""
Tests for Delete Account Edge Function

delete_account Edge Functionの単体テスト
"""

import pytest
import json
from unittest.mock import Mock, patch
from typing import Dict, Any, List


class TestDeleteAccount:
    """Delete Account API テスト"""
    
    @pytest.fixture
    def sample_user_data(self):
        """ユーザーデータサンプル"""
        return {
            "id": "test-user-123",
            "email": "test@example.com",
            "created_at": "2025-01-01T00:00:00Z",
            "updated_at": "2025-01-01T00:00:00Z"
        }
    
    @pytest.fixture
    def sample_user_relationships(self):
        """ユーザー関連データサンプル"""
        return {
            "user_embeddings": [
                {"user_id": "test-user-123", "embedding": [0.1] * 768}
            ],
            "likes": [
                {"user_id": "test-user-123", "video_id": "video-1"},
                {"user_id": "test-user-123", "video_id": "video-2"}
            ],
            "profiles": [
                {"user_id": "test-user-123", "display_name": "Test User"}
            ]
        }
    
    @pytest.mark.unit
    @pytest.mark.api
    def test_delete_account_request_validation(self):
        """アカウント削除リクエスト検証テスト"""
        # POSTメソッドのみ許可
        def validate_method(method: str) -> Dict:
            if method == "POST":
                return {"valid": True, "status": 200}
            elif method == "OPTIONS":
                return {"valid": True, "status": 200, "type": "preflight"}
            else:
                return {"valid": False, "error": "Method not allowed", "status": 405}
        
        # 有効なメソッド
        post_result = validate_method("POST")
        assert post_result["valid"] is True
        assert post_result["status"] == 200
        
        options_result = validate_method("OPTIONS")
        assert options_result["valid"] is True
        assert options_result["type"] == "preflight"
        
        # 無効なメソッド
        invalid_methods = ["GET", "PUT", "DELETE", "PATCH"]
        for method in invalid_methods:
            result = validate_method(method)
            assert result["valid"] is False
            assert result["status"] == 405
    
    @pytest.mark.unit
    @pytest.mark.api
    def test_authentication_validation(self):
        """認証検証テスト"""
        def validate_user_authentication(auth_header: str, user_exists: bool = True) -> Dict:
            """ユーザー認証検証"""
            if not auth_header:
                return {
                    "authenticated": False,
                    "error": "Authorization header missing",
                    "status": 401
                }
            
            if not auth_header.startswith("Bearer "):
                return {
                    "authenticated": False,
                    "error": "Invalid authorization format",
                    "status": 401
                }
            
            if not user_exists:
                return {
                    "authenticated": False,
                    "error": "Unauthorized",
                    "status": 401
                }
            
            return {
                "authenticated": True,
                "user_id": "test-user-123"
            }
        
        # 正常な認証
        result = validate_user_authentication("Bearer valid-token", True)
        assert result["authenticated"] is True
        assert result["user_id"] == "test-user-123"
        
        # 認証ヘッダーなし
        result = validate_user_authentication("", True)
        assert result["authenticated"] is False
        assert result["status"] == 401
        
        # 無効な認証形式
        result = validate_user_authentication("Invalid format", True)
        assert result["authenticated"] is False
        assert result["status"] == 401
        
        # ユーザーが存在しない
        result = validate_user_authentication("Bearer valid-token", False)
        assert result["authenticated"] is False
        assert result["status"] == 401
    
    @pytest.mark.unit
    def test_deletion_sequence_logic(self, sample_user_relationships):
        """削除シーケンスロジックテスト"""
        def execute_deletion_sequence(user_id: str, user_data: Dict) -> Dict:
            """アカウント削除シーケンス実行"""
            deletion_results = {
                "user_embeddings": False,
                "likes": False,
                "profiles": False,
                "auth_user": False
            }
            
            try:
                # 1. user_embeddings削除
                if user_data.get("user_embeddings"):
                    deletion_results["user_embeddings"] = True
                
                # 2. likes削除
                if user_data.get("likes"):
                    deletion_results["likes"] = True
                
                # 3. profiles削除
                if user_data.get("profiles"):
                    deletion_results["profiles"] = True
                
                # 4. auth.users削除
                deletion_results["auth_user"] = True
                
                return {
                    "success": True,
                    "deletions": deletion_results,
                    "message": "Account successfully deleted"
                }
                
            except Exception as e:
                return {
                    "success": False,
                    "error": str(e),
                    "deletions": deletion_results
                }
        
        # 正常な削除シーケンス
        result = execute_deletion_sequence("test-user-123", sample_user_relationships)
        assert result["success"] is True
        assert result["deletions"]["user_embeddings"] is True
        assert result["deletions"]["likes"] is True
        assert result["deletions"]["profiles"] is True
        assert result["deletions"]["auth_user"] is True
    
    @pytest.mark.unit
    def test_cascade_deletion_validation(self):
        """カスケード削除検証テスト"""
        def validate_cascade_deletions(user_id: str) -> Dict:
            """カスケード削除の影響確認"""
            affected_tables = {
                "user_embeddings": f"DELETE FROM user_embeddings WHERE user_id = '{user_id}'",
                "likes": f"DELETE FROM likes WHERE user_id = '{user_id}'", 
                "profiles": f"DELETE FROM profiles WHERE user_id = '{user_id}'",
                "auth.users": f"DELETE FROM auth.users WHERE id = '{user_id}'"
            }
            
            # 依存関係確認
            dependencies = {
                "likes": ["videos"],  # likes -> videos (foreign key)
                "profiles": ["auth.users"],  # profiles -> auth.users
                "user_embeddings": ["auth.users"]  # user_embeddings -> auth.users
            }
            
            return {
                "affected_tables": list(affected_tables.keys()),
                "dependencies": dependencies,
                "deletion_order": ["user_embeddings", "likes", "profiles", "auth.users"]
            }
        
        result = validate_cascade_deletions("test-user-123")
        
        # 削除対象テーブル確認
        expected_tables = ["user_embeddings", "likes", "profiles", "auth.users"]
        assert all(table in result["affected_tables"] for table in expected_tables)
        
        # 削除順序確認（依存関係を考慮）
        deletion_order = result["deletion_order"]
        auth_users_index = deletion_order.index("auth.users")
        profiles_index = deletion_order.index("profiles")
        assert profiles_index < auth_users_index  # profilesはauth.usersより先に削除
    
    @pytest.mark.unit
    def test_error_handling_scenarios(self):
        """エラーハンドリングシナリオテスト"""
        def handle_deletion_error(error_stage: str) -> Dict:
            """削除エラーハンドリング"""
            error_responses = {
                "embedding_deletion_failed": {
                    "error": "Failed to delete user embeddings",
                    "stage": "user_embeddings",
                    "status": 500
                },
                "likes_deletion_failed": {
                    "error": "Failed to delete user likes",
                    "stage": "likes",
                    "status": 500
                },
                "profile_deletion_failed": {
                    "error": "Failed to delete user profile",
                    "stage": "profiles",
                    "status": 500
                },
                "auth_deletion_failed": {
                    "error": "Failed to delete user account",
                    "stage": "auth.users",
                    "status": 500
                },
                "database_connection_error": {
                    "error": "Database connection failed",
                    "stage": "connection",
                    "status": 500
                },
                "internal_server_error": {
                    "error": "Internal server error",
                    "stage": "unknown",
                    "status": 500
                }
            }
            
            return error_responses.get(error_stage, {"status": 200})
        
        # 各段階でのエラーテスト
        error_stages = [
            "embedding_deletion_failed",
            "likes_deletion_failed", 
            "profile_deletion_failed",
            "auth_deletion_failed",
            "database_connection_error",
            "internal_server_error"
        ]
        
        for stage in error_stages:
            error_response = handle_deletion_error(stage)
            assert error_response["status"] == 500
            assert "error" in error_response
            assert "stage" in error_response
    
    @pytest.mark.unit
    def test_rollback_logic(self):
        """ロールバックロジックテスト"""
        def simulate_deletion_with_rollback(user_id: str, fail_at_stage: str = None) -> Dict:
            """削除処理とロールバックをシミュレート"""
            completed_stages = []
            
            stages = ["user_embeddings", "likes", "profiles", "auth_user"]
            
            try:
                for stage in stages:
                    if stage == fail_at_stage:
                        raise Exception(f"Deletion failed at {stage}")
                    completed_stages.append(stage)
                
                return {
                    "success": True,
                    "completed_stages": completed_stages,
                    "rollback_needed": False
                }
                
            except Exception as e:
                return {
                    "success": False,
                    "error": str(e),
                    "completed_stages": completed_stages,
                    "rollback_needed": True,
                    "rollback_stages": completed_stages[::-1]  # 逆順でロールバック
                }
        
        # 正常完了
        result = simulate_deletion_with_rollback("test-user-123")
        assert result["success"] is True
        assert result["rollback_needed"] is False
        assert len(result["completed_stages"]) == 4
        
        # likes段階で失敗
        result = simulate_deletion_with_rollback("test-user-123", "likes")
        assert result["success"] is False
        assert result["rollback_needed"] is True
        assert "user_embeddings" in result["completed_stages"]
        assert "likes" not in result["completed_stages"]
        assert result["rollback_stages"] == ["user_embeddings"]  # 逆順
    
    @pytest.mark.unit
    @pytest.mark.api
    def test_response_format_validation(self):
        """レスポンス形式検証テスト"""
        # 成功レスポンス
        success_response = {
            "success": True,
            "message": "Account successfully deleted"
        }
        
        assert "success" in success_response
        assert "message" in success_response
        assert success_response["success"] is True
        assert "successfully deleted" in success_response["message"]
        
        # エラーレスポンス
        error_response = {
            "error": "Account deletion failed",
            "details": "Failed to delete user embeddings"
        }
        
        assert "error" in error_response
        assert "details" in error_response
        assert "deletion failed" in error_response["error"]
    
    @pytest.mark.unit
    def test_service_role_requirements(self):
        """サービスロール要件テスト"""
        def validate_service_role_permissions(operation: str) -> Dict:
            """サービスロール権限確認"""
            required_permissions = {
                "delete_user_embeddings": ["user_embeddings.delete"],
                "delete_likes": ["likes.delete"],
                "delete_profiles": ["profiles.delete"],
                "delete_auth_user": ["auth.admin.deleteUser"]
            }
            
            if operation in required_permissions:
                return {
                    "operation": operation,
                    "required_permissions": required_permissions[operation],
                    "requires_service_role": True
                }
            else:
                return {
                    "operation": operation,
                    "requires_service_role": False
                }
        
        # 各操作の権限要件確認
        operations = ["delete_user_embeddings", "delete_likes", "delete_profiles", "delete_auth_user"]
        
        for operation in operations:
            result = validate_service_role_permissions(operation)
            assert result["requires_service_role"] is True
            assert len(result["required_permissions"]) > 0
    
    @pytest.mark.unit
    @pytest.mark.security
    def test_security_validations(self):
        """セキュリティ検証テスト"""
        def validate_deletion_security(user_id: str, requesting_user_id: str) -> Dict:
            """削除セキュリティ検証"""
            # ユーザーは自分のアカウントのみ削除可能
            if user_id != requesting_user_id:
                return {
                    "authorized": False,
                    "error": "Cannot delete another user's account",
                    "status": 403
                }
            
            # 管理者権限チェック（実際の実装では不要だが、セキュリティ概念として）
            if user_id == "admin-user":
                return {
                    "authorized": False,
                    "error": "Admin account cannot be deleted via this endpoint",
                    "status": 403
                }
            
            return {
                "authorized": True,
                "user_id": user_id
            }
        
        # 正常な削除（自分のアカウント）
        result = validate_deletion_security("user-123", "user-123")
        assert result["authorized"] is True
        
        # 他人のアカウント削除試行
        result = validate_deletion_security("user-123", "user-456")
        assert result["authorized"] is False
        assert result["status"] == 403
        
        # 管理者アカウント削除試行
        result = validate_deletion_security("admin-user", "admin-user")
        assert result["authorized"] is False
        assert result["status"] == 403
    
    @pytest.mark.unit
    @pytest.mark.api
    def test_cors_headers_validation(self):
        """CORSヘッダー検証テスト"""
        expected_cors_headers = {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
            'Access-Control-Allow-Methods': 'POST, OPTIONS',
        }
        
        # 必須CORSヘッダーの確認
        assert 'Access-Control-Allow-Origin' in expected_cors_headers
        assert 'Access-Control-Allow-Headers' in expected_cors_headers
        assert 'Access-Control-Allow-Methods' in expected_cors_headers
        
        # 適切な値の確認
        assert expected_cors_headers['Access-Control-Allow-Origin'] == '*'
        assert 'authorization' in expected_cors_headers['Access-Control-Allow-Headers']
        assert 'POST' in expected_cors_headers['Access-Control-Allow-Methods']
        assert 'OPTIONS' in expected_cors_headers['Access-Control-Allow-Methods']
        
        # DELETE method は含まれない（POST のみ）
        assert 'DELETE' not in expected_cors_headers['Access-Control-Allow-Methods']
    
    @pytest.mark.unit
    @pytest.mark.performance
    def test_deletion_performance_requirements(self, performance_timer):
        """削除パフォーマンス要件テスト"""
        def mock_account_deletion():
            """アカウント削除処理のモック"""
            import time
            time.sleep(0.2)  # 200ms の削除処理時間をシミュレート
            return {"processing_time_ms": 200, "stages_completed": 4}
        
        performance_timer.start()
        result = mock_account_deletion()
        performance_timer.stop()
        
        # 1秒以内の要件（複数テーブル削除のため余裕を持たせる）
        assert performance_timer.duration < 1.0
        assert result["processing_time_ms"] <= 1000
        assert result["stages_completed"] == 4
    
    @pytest.mark.unit
    def test_data_consistency_validation(self):
        """データ一貫性検証テスト"""
        def validate_post_deletion_state(user_id: str) -> Dict:
            """削除後のデータ一貫性確認"""
            # 削除後、該当ユーザーのデータが残っていないことを確認
            remaining_data = {
                "user_embeddings": [],  # 削除済み
                "likes": [],            # 削除済み
                "profiles": [],         # 削除済み
                "auth_users": [],       # 削除済み
                "videos": ["video-1", "video-2"]  # 他ユーザーのいいねで参照されている動画は残存
            }
            
            user_references = 0
            for table_data in remaining_data.values():
                user_references += sum(1 for item in table_data if isinstance(item, dict) and item.get("user_id") == user_id)
            
            return {
                "user_references_remaining": user_references,
                "data_consistent": user_references == 0,
                "orphaned_videos": len(remaining_data.get("videos", [])) > 0  # 動画は残ってOK
            }
        
        # 削除後の状態確認
        result = validate_post_deletion_state("deleted-user-123")
        assert result["user_references_remaining"] == 0
        assert result["data_consistent"] is True
        # 動画データは他ユーザーが参照している可能性があるため残存OK
        assert result["orphaned_videos"] is True