from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


if __name__ == "__main__":
    print("assets", client.get("/api/assets/health").status_code)
    print("translate", client.post("/api/translation/text", json={"text": "I am going to university"}).status_code)
    print(
        "render",
        client.post(
            "/api/render/sequence",
            json={"tokens": ["i", "university", "go"], "render_mode": "animated_video"},
        ).status_code,
    )
