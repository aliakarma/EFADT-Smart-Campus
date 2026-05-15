"""
EFADT — API Integration Tests
================================
Tests the FastAPI endpoints using httpx async client.
"""

import pytest
import numpy as np

pytest_plugins = ("anyio",)


@pytest.fixture
def client():
    """Create synchronous test client."""
    from fastapi.testclient import TestClient
    from api.main import app
    with TestClient(app) as c:
        yield c


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "healthy"


def test_status(client):
    resp = client.get("/status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["n_buildings"] > 0


def test_list_buildings(client):
    resp = client.get("/buildings")
    assert resp.status_code == 200
    buildings = resp.json()["buildings"]
    assert len(buildings) >= 1
    assert "B01" in buildings


def test_decide_endpoint(client):
    payload = {
        "building_id": "B01",
        "sensor": {
            "building_id": "B01",
            "temperature_in": 25.5,
            "temperature_out": 38.0,
            "occupancy": 45.0,
            "co2_ppm": 650.0,
            "humidity": 55.0,
            "hvac_power_kw": -8.0,
            "hvac_setpoint": 22.0,
            "motion_count": 12,
        },
        "occ_forecast": [45.0, 48.0, 50.0, 52.0, 50.0, 47.0],
    }
    resp = client.post("/decide", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "hvac_power_kw" in data
    assert "trust_score" in data
    assert 0.0 <= data["trust_score"] <= 1.0
    assert "top_features" in data


def test_decide_unknown_building(client):
    payload = {
        "building_id": "UNKNOWN",
        "sensor": {
            "building_id": "UNKNOWN",
            "temperature_in": 22.0,
            "temperature_out": 30.0,
            "occupancy": 20.0,
            "co2_ppm": 550.0,
            "humidity": 45.0,
        },
    }
    resp = client.post("/decide", json=payload)
    assert resp.status_code == 404


def test_simulate_endpoint(client):
    resp = client.get("/simulate/B01?T_in=26.0&T_out=39.0&Q_hvac=-15.0&occ=60.0&H=6")
    assert resp.status_code == 200
    data = resp.json()
    assert "T_trajectory" in data
    assert len(data["T_trajectory"]) == 6
    assert "E" in data and "C" in data and "D" in data


def test_metrics_summary(client):
    resp = client.get("/metrics/summary")
    assert resp.status_code == 200
    data = resp.json()
    assert "EFADT (Full)" in data["paper_results"]


def test_audit_log_empty(client):
    resp = client.get("/audit/B01?max_records=10")
    assert resp.status_code == 200
    data = resp.json()
    assert "records" in data


def test_audit_chain_verify(client):
    resp = client.get("/audit/B01/verify")
    assert resp.status_code == 200
    data = resp.json()
    assert "chain_valid" in data
    assert data["chain_valid"] is True
