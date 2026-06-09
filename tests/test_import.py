from pathlib import Path
import importlib.util


def test_analyzer_module_can_be_loaded():
    module_path = Path(__file__).resolve().parents[1] / "fitness_analyzer (2).py"
    spec = importlib.util.spec_from_file_location("fitness_analyzer_module", module_path)

    assert spec is not None and spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert hasattr(module, "EnhancedFitnessAnalyzer")
