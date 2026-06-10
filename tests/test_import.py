from pathlib import Path
import importlib.util


def test_analyzer_module_can_be_loaded():
    module_path = Path(__file__).resolve().parents[1] / "fitness_analyzer (2).py"
    spec = importlib.util.spec_from_file_location("fitness_analyzer_module", module_path)

    assert spec is not None and spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert hasattr(module, "EnhancedFitnessAnalyzer")


def test_analyzer_initializes_when_mediapipe_is_unavailable(monkeypatch):
    module_path = Path(__file__).resolve().parents[1] / "fitness_analyzer (2).py"
    spec = importlib.util.spec_from_file_location("fitness_analyzer_module", module_path)

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "mp", None)

    analyzer = module.EnhancedFitnessAnalyzer()

    assert analyzer.pose is None
    assert analyzer.mp_pose is None
