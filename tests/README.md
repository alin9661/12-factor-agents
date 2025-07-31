# 12-Factor Agents Test Suite

This comprehensive test suite follows test-driven development (TDD) principles to ensure the reliability, security, and performance of the 12-factor agents system.

## Test Categories

### 🧪 Unit Tests (`tests/unit/`)
Test individual components in isolation:
- **Tool Registry** (`test_tool_registry.py`) - Tool registration, execution, validation
- **Base Agent** (`test_base_agent.py`) - Agent execution logic, stateless reducer behavior  
- **LLM Providers** (`test_llm_providers.py`) - OpenAI/Anthropic provider abstractions
- **Built-in Tools** (`test_builtin_tools.py`) - All 11 built-in tools functionality

### 🔗 Integration Tests (`tests/integration/`)
Test API endpoints and external system interactions:
- **API Endpoints** (`test_api_endpoints.py`) - FastAPI route testing with real HTTP requests
- **Error Handling** - API error responses and edge cases
- **CORS and Authentication** - Cross-origin and auth behavior

### 🛡️ Security Tests (`tests/security/`)
Identify and demonstrate security vulnerabilities:
- **Path Traversal** - File operation vulnerabilities (`../../../etc/passwd`)
- **Input Validation** - SQL injection, XSS, command injection
- **Authentication** - Missing auth/authorization checks  
- **Data Exposure** - Sensitive information leakage
- **Session Management** - Session security issues

⚠️ **IMPORTANT**: Security tests are designed to PASS initially (demonstrating vulnerabilities exist). After fixing security issues, these tests should FAIL (showing vulnerabilities are patched).

### 🔄 End-to-End Tests (`tests/e2e/`)
Test complete workflows from start to finish:
- **Deployment Workflows** - Full agent deployment scenarios
- **Customer Support** - Complete ticket creation workflows
- **Webhook Integration** - External system triggered workflows
- **Multi-Agent** - Concurrent session management

### ⚡ Performance Tests (`tests/performance/`)
Test system performance and scalability:
- **API Performance** - Response times, throughput
- **Load Testing** - Concurrent request handling
- **Memory Usage** - Resource consumption patterns
- **Stress Testing** - Breaking point identification

## Test Infrastructure

### Mock Classes (`tests/mocks/`)
- **LLM Providers** - Mock OpenAI/Anthropic for deterministic testing
- **External Services** - HTTP, file system, email, deployment services
- **Configurable Responses** - Predefined response sequences for workflows

### Fixtures (`tests/conftest.py`)
- **Agent Contexts** - Sample conversation history and session data
- **Test Clients** - FastAPI TestClient and AsyncClient instances
- **Security Payloads** - Malicious inputs for vulnerability testing
- **Performance Data** - Load testing configurations

## Running Tests

### Quick Start
```bash
# Run all fast tests (excludes slow/performance tests)
python run_tests.py fast

# Run specific test category
python run_tests.py unit
python run_tests.py integration
python run_tests.py security

# Run with coverage report
python run_tests.py unit --coverage

# Run with HTML reports
python run_tests.py integration --html-report
```

### Using pytest directly
```bash
# Run specific test categories by marker
pytest -m unit                    # Unit tests only
pytest -m integration            # Integration tests only  
pytest -m security              # Security tests only
pytest -m "not slow"           # Exclude slow tests
pytest -m "performance"        # Performance tests only

# Run specific test files
pytest tests/unit/test_tool_registry.py
pytest tests/security/test_vulnerabilities.py -v

# Run with coverage
pytest --cov=src --cov-report=html tests/unit/

# Run in parallel
pytest -n auto tests/unit/
```

### Test Configuration
Configuration is in `pyproject.toml`:
```toml
[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
addopts = [
    "--cov=src",
    "--cov-report=term-missing", 
    "--cov-report=html:htmlcov",
    "--cov-branch",
    "--cov-fail-under=80"
]
```

## Test Markers

- `@pytest.mark.unit` - Unit tests for individual components
- `@pytest.mark.integration` - Integration tests with external systems
- `@pytest.mark.security` - Security vulnerability tests
- `@pytest.mark.performance` - Performance and load tests  
- `@pytest.mark.e2e` - End-to-end workflow tests
- `@pytest.mark.slow` - Tests that take a long time to run

## Expected Test Results (TDD Approach)

### ✅ Should Pass Initially
- Most unit tests (basic functionality)
- Integration tests (API structure)
- Performance baseline tests

### ❌ Should Fail Initially (By Design)
- **Security vulnerability tests** - Demonstrate security issues exist
- **Advanced functionality tests** - Features not yet implemented
- **Error handling tests** - Proper error handling not implemented
- **Authentication tests** - No auth system implemented

### 🔄 Implementation Cycle
1. **Red** - Write failing test that describes desired behavior
2. **Green** - Implement minimal code to make test pass
3. **Refactor** - Improve code while keeping tests green

## Security Test Warnings

The security tests in `tests/security/` are designed to expose real vulnerabilities:

- **Path Traversal**: Tests attempt to access `../../../etc/passwd`
- **SQL Injection**: Tests inject SQL commands into input fields
- **XSS**: Tests inject JavaScript payloads
- **Command Injection**: Tests attempt to execute system commands

⚠️ **These tests should initially PASS (indicating vulnerabilities exist).** After implementing security fixes, these tests should FAIL (indicating vulnerabilities are blocked).

## Coverage Goals

- **Overall Coverage**: 80%+ 
- **Unit Tests**: 90%+ coverage of core logic
- **Integration Tests**: 100% of API endpoints
- **Security Tests**: 100% of identified vulnerability classes

## Performance Benchmarks

- **Health Endpoint**: < 50ms average response time
- **Agent Creation**: < 500ms average response time  
- **Concurrent Requests**: > 100 RPS for health endpoint
- **Session Queries**: < 100ms average response time
- **Memory Usage**: Proper cleanup of deleted sessions

## Continuous Integration

For CI environments, use:
```bash
python run_tests.py ci
```

This runs:
- All non-slow tests
- Excludes performance tests
- Fails fast (maxfail=5)
- Short traceback format
- Generates coverage reports

## Adding New Tests

### 1. Choose Test Category
- **Unit**: Testing individual functions/classes
- **Integration**: Testing API endpoints
- **Security**: Testing for vulnerabilities
- **E2E**: Testing complete workflows
- **Performance**: Testing speed/throughput

### 2. Use Appropriate Fixtures
```python
def test_something(async_client, sample_agent_context, mock_llm_provider):
    # Test implementation
```

### 3. Follow TDD Process
1. Write failing test first
2. Implement minimal code to pass
3. Refactor while keeping tests green

### 4. Add Appropriate Markers
```python
@pytest.mark.unit
@pytest.mark.asyncio
async def test_new_feature():
    # Test implementation
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure `PYTHONPATH` includes project root
2. **Async Errors**: Use `@pytest.mark.asyncio` for async tests
3. **Mock Errors**: Check mock provider configuration
4. **Security Test Confusion**: Security tests SHOULD pass initially

### Debug Mode
```bash
# Run single test with full output
pytest tests/unit/test_tool_registry.py::test_specific_function -v -s

# Run with debugger on failure
pytest --pdb tests/unit/test_tool_registry.py

# Run with warnings shown
pytest --disable-warnings tests/unit/
```

## Contributing

1. All new features must have tests written first (TDD)
2. Maintain 80%+ test coverage
3. Security tests must demonstrate vulnerabilities before fixes
4. Performance tests must include benchmarks
5. All tests must pass in CI environment

## Reports

Test reports are generated in:
- **Coverage**: `htmlcov/index.html`
- **Test Results**: `reports/test_report.html`
- **Performance**: Console output with timing statistics

---

This test suite provides comprehensive coverage of the 12-factor agents system, following TDD principles to guide development and ensure production readiness.