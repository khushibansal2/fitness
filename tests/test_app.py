from app import app as main_app
from render_app import app as render_app


def test_health_endpoint_returns_ok():
    client = main_app.test_client()
    response = client.get('/health')

    assert response.status_code == 200
    assert response.get_json() == {'status': 'ok'}


def test_analyze_requires_video_upload():
    client = render_app.test_client()
    response = client.post('/analyze', data={})

    assert response.status_code == 400
    assert response.get_json() == {'error': 'No video file uploaded'}
