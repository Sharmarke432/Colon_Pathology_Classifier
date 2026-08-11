# tests/conftest.py
import pytest
from fastapi.testclient import TestClient

from src.api.main import app


@pytest.fixture
def client() -> TestClient:
    """
    Create a TestClient for the FastAPI app.
    """
    with TestClient(app) as c:
        yield c
