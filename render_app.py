import importlib.util
import json
import os
import tempfile
from pathlib import Path

from flask import Flask, jsonify, render_template_string, request

app = Flask(__name__)
MODULE_PATH = Path(__file__).resolve().with_name("fitness_analyzer (2).py")


HTML_TEMPLATE = """
<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\">
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
  <title>Fitness Analyzer</title>
  <style>
    body { font-family: Arial, sans-serif; max-width: 900px; margin: 2rem auto; padding: 0 1rem; }
    form { background: #f7f7f7; padding: 1rem; border-radius: 8px; }
    input, select, button { display: block; width: 100%; margin: 0.5rem 0 1rem; padding: 0.6rem; }
    pre { background: #111; color: #eee; padding: 1rem; overflow-x: auto; }
  </style>
</head>
<body>
  <h1>Fitness Analyzer</h1>
  <p>Upload a video to analyze sit-ups, vertical jump, broad jump, or flexibility.</p>
  <form action=\"/analyze\" method=\"post\" enctype=\"multipart/form-data\">
    <label>Video file</label>
    <input type=\"file\" name=\"video\" required accept=\".mp4,.mov,.avi,.mkv\">
    <label>Test type</label>
    <select name=\"test_type\">
      <option value=\"situps\">Sit-ups</option>
      <option value=\"vertical_jump\">Vertical jump</option>
      <option value=\"broad_jump\">Broad jump</option>
      <option value=\"flexibility\">Flexibility</option>
    </select>
    <label>Age group</label>
    <select name=\"age_group\">
      <option value=\"teenage\">Teenage</option>
      <option value=\"youth\">Youth</option>
      <option value=\"adult\">Adult</option>
    </select>
    <label>Gender</label>
    <select name=\"gender\">
      <option value=\"male\">Male</option>
      <option value=\"female\">Female</option>
      <option value=\"other\">Other</option>
    </select>
    <button type=\"submit\">Run analysis</button>
  </form>
  {% if result_json %}
    <h2>Results</h2>
    <pre>{{ result_json }}</pre>
  {% endif %}
</body>
</html>
"""


def load_analyzer():
    spec = importlib.util.spec_from_file_location("fitness_analyzer_module", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load the fitness analyzer module")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.EnhancedFitnessAnalyzer()


@app.get('/')
def index():
    return render_template_string(HTML_TEMPLATE, result_json=None)


@app.get('/health')
def health():
    return jsonify({'status': 'ok'})


@app.post('/analyze')
def analyze():
    video_file = request.files.get('video')
    if not video_file or not video_file.filename:
        return jsonify({'error': 'No video file uploaded'}), 400

    test_type = request.form.get('test_type', 'situps')
    age_group = request.form.get('age_group', 'adult')
    gender = request.form.get('gender', 'other')

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(video_file.filename).suffix or '.mp4') as temp_file:
            temp_file.write(video_file.read())
            temp_path = temp_file.name

        analyzer = load_analyzer()
        if test_type == 'situps':
            result = analyzer.analyze_situps(temp_path, age_group, gender, False)
        elif test_type == 'vertical_jump':
            result = analyzer.analyze_vertical_jump(temp_path, age_group, gender, False)
        elif test_type == 'broad_jump':
            result = analyzer.analyze_broad_jump(temp_path, age_group, gender, False)
        else:
            result = analyzer.analyze_flexibility(temp_path, age_group, gender, False)

        if os.path.exists(temp_path):
            os.remove(temp_path)

        return render_template_string(HTML_TEMPLATE, result_json=json.dumps(result, indent=2))
    except Exception as exc:
        return jsonify({'error': str(exc)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
