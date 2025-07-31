"""
Performance and load tests for the 12-factor agents API.

These tests verify system performance under various load conditions
and help identify bottlenecks and scalability issues.
"""

import asyncio
import time
import statistics
import pytest
from typing import List, Dict, Any, Tuple
from unittest.mock import patch

from httpx import AsyncClient
from fastapi import status

from src.api.main import agent_sessions


@pytest.mark.performance
@pytest.mark.slow
class TestAPIPerformance:
    """Test API endpoint performance under various conditions."""
    
    def setup_method(self):
        """Clear agent sessions before each test."""
        agent_sessions.clear()
    
    @pytest.mark.asyncio
    async def test_health_endpoint_response_time(self, async_client: AsyncClient):
        """Test health endpoint response time."""
        
        response_times = []
        iterations = 100
        
        for _ in range(iterations):
            start_time = time.time()
            response = await async_client.get("/health")
            end_time = time.time()
            
            assert response.status_code == status.HTTP_200_OK
            response_times.append((end_time - start_time) * 1000)  # Convert to milliseconds
        
        # Performance assertions  
        avg_response_time = statistics.mean(response_times)
        max_response_time = max(response_times)
        p95_response_time = statistics.quantiles(response_times, n=20)[18]  # 95th percentile
        
        # Health endpoint should be very fast
        assert avg_response_time < 50, f"Average response time {avg_response_time:.2f}ms exceeds 50ms threshold"
        assert max_response_time < 200, f"Max response time {max_response_time:.2f}ms exceeds 200ms threshold"  
        assert p95_response_time < 100, f"95th percentile {p95_response_time:.2f}ms exceeds 100ms threshold"
    
    @pytest.mark.asyncio
    async def test_concurrent_health_requests(self, async_client: AsyncClient):
        """Test health endpoint under concurrent load."""
        
        concurrent_requests = 50
        start_time = time.time()
        
        # Create concurrent requests
        tasks = [async_client.get("/health") for _ in range(concurrent_requests)]
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Verify all requests succeeded
        successful_responses = 0
        for response in responses:
            if not isinstance(response, Exception) and response.status_code == status.HTTP_200_OK:
                successful_responses += 1
        
        assert successful_responses == concurrent_requests, f"Only {successful_responses}/{concurrent_requests} requests succeeded"
        
        # Performance assertions
        requests_per_second = concurrent_requests / total_time
        assert requests_per_second > 100, f"RPS {requests_per_second:.2f} below threshold of 100"
        assert total_time < 5.0, f"Total time {total_time:.2f}s exceeds 5s threshold"
    
    @pytest.mark.asyncio
    async def test_agent_creation_performance(self, async_client: AsyncClient, performance_test_data: Dict[str, Any]):
        """Test agent creation endpoint performance."""
        
        with patch('src.api.main.DeploymentAgent'):
            with patch('src.api.main.run_agent_background'):
                
                response_times = []
                successful_requests = 0
                
                # Test sequential agent creation
                for i in range(20):
                    request_data = {
                        "message": f"Performance test {i}",
                        "agent_type": "deployment",
                        "user_id": f"perf_user_{i}"
                    }
                    
                    start_time = time.time()
                    response = await async_client.post("/agents/start", json=request_data)
                    end_time = time.time()
                    
                    if response.status_code == status.HTTP_200_OK:
                        successful_requests += 1
                        response_times.append((end_time - start_time) * 1000)
                
                assert successful_requests == 20, f"Only {successful_requests}/20 agent creations succeeded"
                
                # Performance assertions
                avg_response_time = statistics.mean(response_times)
                max_response_time = max(response_times)
                
                assert avg_response_time < 500, f"Average agent creation time {avg_response_time:.2f}ms exceeds 500ms"
                assert max_response_time < 2000, f"Max agent creation time {max_response_time:.2f}ms exceeds 2s"
    
    @pytest.mark.asyncio
    async def test_concurrent_agent_creation(self, async_client: AsyncClient):
        """Test concurrent agent creation performance."""
        
        with patch('src.api.main.DeploymentAgent'):
            with patch('src.api.main.run_agent_background'):
                
                concurrent_requests = 20
                start_time = time.time()
                
                # Create concurrent agent requests
                tasks = []
                for i in range(concurrent_requests):
                    request_data = {
                        "message": f"Concurrent test {i}",
                        "agent_type": "deployment",
                        "user_id": f"concurrent_user_{i}"
                    }
                    tasks.append(async_client.post("/agents/start", json=request_data))
                
                responses = await asyncio.gather(*tasks, return_exceptions=True)
                end_time = time.time()
                
                total_time = end_time - start_time
                
                # Count successful responses
                successful_responses = 0
                for response in responses:
                    if not isinstance(response, Exception) and response.status_code == status.HTTP_200_OK:
                        successful_responses += 1
                
                assert successful_responses >= concurrent_requests * 0.9, f"Too many failures: {successful_responses}/{concurrent_requests}"
                
                # Performance assertions
                requests_per_second = successful_responses / total_time
                assert requests_per_second > 5, f"Agent creation RPS {requests_per_second:.2f} below threshold"
                assert total_time < 10.0, f"Total time {total_time:.2f}s exceeds 10s threshold"
    
    @pytest.mark.asyncio
    async def test_session_status_query_performance(self, async_client: AsyncClient):
        """Test performance of querying agent session status."""
        
        with patch('src.api.main.DeploymentAgent'):
            with patch('src.api.main.run_agent_background'):
                
                # Create several agent sessions first
                thread_ids = []
                for i in range(10):
                    request_data = {
                        "message": f"Status test {i}",
                        "agent_type": "deployment"
                    }
                    response = await async_client.post("/agents/start", json=request_data)
                    thread_ids.append(response.json()["thread_id"])
                
                # Test status query performance
                response_times = []
                for thread_id in thread_ids:
                    start_time = time.time()
                    response = await async_client.get(f"/agents/{thread_id}/status")
                    end_time = time.time()
                    
                    assert response.status_code == status.HTTP_200_OK
                    response_times.append((end_time - start_time) * 1000)
                
                # Performance assertions
                avg_response_time = statistics.mean(response_times)
                max_response_time = max(response_times)
                
                assert avg_response_time < 100, f"Average status query time {avg_response_time:.2f}ms exceeds 100ms"
                assert max_response_time < 300, f"Max status query time {max_response_time:.2f}ms exceeds 300ms"
    
    @pytest.mark.asyncio
    async def test_list_sessions_performance_with_scale(self, async_client: AsyncClient):
        """Test list sessions performance with many sessions."""
        
        with patch('src.api.main.DeploymentAgent'):
            with patch('src.api.main.run_agent_background'):
                
                # Create many sessions to test scaling
                session_counts = [10, 50, 100]
                
                for session_count in session_counts:
                    # Clear previous sessions
                    agent_sessions.clear()
                    
                    # Create sessions
                    for i in range(session_count):
                        request_data = {
                            "message": f"Scale test {i}",
                            "agent_type": "deployment",
                            "user_id": f"scale_user_{i}"
                        }
                        await async_client.post("/agents/start", json=request_data)
                    
                    # Test list performance
                    start_time = time.time()
                    response = await async_client.get("/agents")
                    end_time = time.time()
                    
                    response_time = (end_time - start_time) * 1000
                    
                    assert response.status_code == status.HTTP_200_OK
                    data = response.json()
                    assert data["total_count"] == session_count
                    
                    # Performance should not degrade significantly with more sessions
                    max_time = 50 + (session_count * 0.5)  # Allow 0.5ms per session plus base 50ms
                    assert response_time < max_time, f"List {session_count} sessions took {response_time:.2f}ms (max {max_time:.2f}ms)"


@pytest.mark.performance
class TestMemoryUsage:
    """Test memory usage and resource management."""
    
    def setup_method(self):
        """Clear agent sessions before each test."""
        agent_sessions.clear()
    
    @pytest.mark.asyncio
    async def test_session_memory_usage(self, async_client: AsyncClient):
        """Test that session storage doesn't grow unbounded."""
        
        with patch('src.api.main.DeploymentAgent'):
            with patch('src.api.main.run_agent_background'):
                
                initial_session_count = len(agent_sessions)
                
                # Create and delete many sessions
                created_sessions = []
                for i in range(50):
                    request_data = {
                        "message": f"Memory test {i}",
                        "agent_type": "deployment"
                    }
                    response = await async_client.post("/agents/start", json=request_data)
                    thread_id = response.json()["thread_id"]
                    created_sessions.append(thread_id)
                
                # Verify sessions were created
                assert len(agent_sessions) == initial_session_count + 50
                
                # Delete all sessions
                for thread_id in created_sessions:
                    await async_client.delete(f"/agents/{thread_id}")
                
                # Verify sessions were deleted
                assert len(agent_sessions) == initial_session_count
    
    @pytest.mark.asyncio
    async def test_large_session_data_handling(self, async_client: AsyncClient):
        """Test handling of sessions with large amounts of data."""
        
        with patch('src.api.main.DeploymentAgent'):
            with patch('src.api.main.run_agent_background'):
                
                # Create session with large data
                large_data = "x" * (1024 * 10)  # 10KB of data
                request_data = {
                    "message": "Large data test",
                    "agent_type": "deployment",
                    "session_data": {
                        "large_field": large_data,
                        "metadata": {"size": len(large_data)}
                    }
                }
                
                start_time = time.time()
                response = await async_client.post("/agents/start", json=request_data)
                creation_time = (time.time() - start_time) * 1000
                
                assert response.status_code == status.HTTP_200_OK
                thread_id = response.json()["thread_id"]
                
                # Performance should not degrade significantly with large data
                assert creation_time < 1000, f"Large session creation took {creation_time:.2f}ms"
                
                # Test status query with large session
                start_time = time.time()
                status_response = await async_client.get(f"/agents/{thread_id}/status")
                status_time = (time.time() - start_time) * 1000
                
                assert status_response.status_code == status.HTTP_200_OK
                assert status_time < 500, f"Large session status query took {status_time:.2f}ms"


@pytest.mark.performance
@pytest.mark.slow
class TestStressTests:
    """Stress tests to identify breaking points."""
    
    def setup_method(self):
        """Clear agent sessions before each test."""
        agent_sessions.clear()
    
    @pytest.mark.asyncio
    async def test_rapid_session_creation_and_deletion(self, async_client: AsyncClient):
        """Test rapid creation and deletion of sessions."""
        
        with patch('src.api.main.DeploymentAgent'):
            with patch('src.api.main.run_agent_background'):
                
                iterations = 100
                errors = []
                response_times = []
                
                for i in range(iterations):
                    try:
                        # Create session
                        request_data = {
                            "message": f"Rapid test {i}",
                            "agent_type": "deployment"
                        }
                        
                        start_time = time.time()
                        create_response = await async_client.post("/agents/start", json=request_data)
                        
                        if create_response.status_code == status.HTTP_200_OK:
                            thread_id = create_response.json()["thread_id"]
                            
                            # Immediately delete session
                            delete_response = await async_client.delete(f"/agents/{thread_id}")
                            end_time = time.time()
                            
                            if delete_response.status_code == status.HTTP_200_OK:
                                response_times.append((end_time - start_time) * 1000)
                            else:
                                errors.append(f"Delete {i}: {delete_response.status_code}")
                        else:
                            errors.append(f"Create {i}: {create_response.status_code}")
                            
                    except Exception as e:
                        errors.append(f"Exception {i}: {str(e)}")
                
                # Should handle rapid operations gracefully
                error_rate = len(errors) / iterations
                assert error_rate < 0.05, f"Error rate {error_rate:.2%} exceeds 5%: {errors[:5]}"
                
                if response_times:
                    avg_time = statistics.mean(response_times)
                    assert avg_time < 1000, f"Average rapid operation time {avg_time:.2f}ms exceeds 1s"
    
    @pytest.mark.asyncio
    async def test_webhook_flood_handling(self, async_client: AsyncClient):
        """Test handling of many webhook requests in quick succession."""
        
        with patch('src.api.main.start_agent') as mock_start_agent:
            # Mock successful agent starts
            mock_start_agent.return_value = {
                "thread_id": "mock_thread",
                "agent_type": "deployment", 
                "status": "started",
                "execution_state": "running"
            }
            
            flood_size = 50
            start_time = time.time()
            
            # Send many webhook requests rapidly
            tasks = []
            for i in range(flood_size):
                payload = {
                    "message": f"Flood test {i}",
                    "agent_type": "deployment",
                    "context": {"flood_id": i}
                }
                tasks.append(async_client.post("/webhooks/trigger", json=payload))
            
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            end_time = time.time()
            
            total_time = end_time - start_time
            
            # Count successful responses
            successful_responses = 0
            for response in responses:
                if not isinstance(response, Exception) and response.status_code == status.HTTP_200_OK:
                    successful_responses += 1
            
            # Should handle most requests successfully
            success_rate = successful_responses / flood_size
            assert success_rate > 0.8, f"Success rate {success_rate:.2%} below 80%"
            
            # Should maintain reasonable throughput
            requests_per_second = successful_responses / total_time
            assert requests_per_second > 10, f"Webhook RPS {requests_per_second:.2f} below threshold"
    
    @pytest.mark.asyncio
    async def test_session_query_performance_degradation(self, async_client: AsyncClient):
        """Test how performance degrades with many active sessions."""
        
        with patch('src.api.main.DeploymentAgent'):
            with patch('src.api.main.run_agent_background'):
                
                # Create baseline measurement with few sessions
                baseline_sessions = 5
                for i in range(baseline_sessions):
                    request_data = {"message": f"Baseline {i}", "agent_type": "deployment"}
                    await async_client.post("/agents/start", json=request_data)
                
                start_time = time.time()
                response = await async_client.get("/agents")
                baseline_time = (time.time() - start_time) * 1000
                
                assert response.status_code == status.HTTP_200_OK
                
                # Add many more sessions
                additional_sessions = 95  # Total will be 100
                for i in range(additional_sessions):
                    request_data = {"message": f"Load {i}", "agent_type": "deployment"}
                    await async_client.post("/agents/start", json=request_data)
                
                # Measure performance with many sessions
                start_time = time.time()
                response = await async_client.get("/agents")
                loaded_time = (time.time() - start_time) * 1000
                
                assert response.status_code == status.HTTP_200_OK
                data = response.json()
                assert data["total_count"] == 100
                
                # Performance should not degrade too much
                degradation_factor = loaded_time / baseline_time
                assert degradation_factor < 10, f"Performance degraded {degradation_factor:.1f}x with more sessions"
                assert loaded_time < 1000, f"Query with 100 sessions took {loaded_time:.2f}ms"


@pytest.mark.performance
class TestResourceLimits:
    """Test system resource limits and boundaries."""
    
    @pytest.mark.asyncio
    async def test_maximum_concurrent_connections(self, async_client: AsyncClient):
        """Test behavior at maximum concurrent connections."""
        
        # This test would need to be adjusted based on actual connection limits
        max_connections = 100
        
        with patch('src.api.main.DeploymentAgent'):
            with patch('src.api.main.run_agent_background'):
                
                # Create many simultaneous long-running requests
                async def long_running_request(i):
                    request_data = {
                        "message": f"Long running {i}",
                        "agent_type": "deployment"
                    }
                    return await async_client.post("/agents/start", json=request_data)
                
                tasks = [long_running_request(i) for i in range(max_connections)]
                
                start_time = time.time()
                responses = await asyncio.gather(*tasks, return_exceptions=True)
                end_time = time.time()
                
                # Count successful responses
                successful_responses = 0
                connection_errors = 0
                
                for response in responses:
                    if isinstance(response, Exception):
                        connection_errors += 1
                    elif response.status_code == status.HTTP_200_OK:
                        successful_responses += 1
                
                # Should handle reasonable number of concurrent connections
                success_rate = successful_responses / max_connections
                assert success_rate > 0.7, f"Only {success_rate:.2%} of {max_connections} concurrent requests succeeded"
                
                total_time = end_time - start_time
                assert total_time < 30, f"Processing {max_connections} requests took {total_time:.2f}s"