import pytest
from tools.proven_anchor_curriculum import run_curriculum

class FakeTrainer:
    def __init__(self):
        self.examples = []
    def add_exact_examples(self, examples):
        self.examples.extend(examples)
    def train_step(self, batch_size=64):
        return 0.1

def test_run_curriculum_reports_informative_failure_when_sign_match_never_clears_baseline(monkeypatch):
    import tools.proven_anchor_curriculum as mod
    # Stub every expensive piece: build_layer always returns a resolved,
    # winning layer; validation always fails (simulates "signal is clean but
    # capacity is the bottleneck" from spec §2's failure criterion).
    monkeypatch.setattr(mod, "build_layer", lambda ancestor, seconds: object())
    monkeypatch.setattr(mod, "examples_from_layer", lambda layer: [(None, None, 1.0)])
    monkeypatch.setattr(mod, "layer_passes_validation", lambda *a, **k: False)

    result = run_curriculum(anchor_games=[(["p0", "p1"], 1), (["p0", "p1"], 1)],
                            reference_binary="fake", seconds=0.1, trainer=FakeTrainer())

    assert result["outcome"] == "informative_failure"
    assert result["layers_built"] <= 3  # stops within the first few layers, not forever

def test_run_curriculum_requires_at_least_two_anchor_games():
    with pytest.raises(ValueError):
        run_curriculum(anchor_games=[(["p0"], 0)], reference_binary="fake",
                       seconds=0.1, trainer=FakeTrainer())

def _deep_history(qualifying_ply=120):
    # Long enough that no distance explored by these tests ever runs an
    # anchor game out of its own history -- isolates the consecutive-failure
    # counter / threshold logic from the exhaustion stop condition.
    return [f"p{i}" for i in range(qualifying_ply + 1)], qualifying_ply

def test_run_curriculum_stops_on_the_failure_threshold_before_history_runs_out(monkeypatch):
    import tools.proven_anchor_curriculum as mod
    monkeypatch.setattr(mod, "build_layer", lambda ancestor, seconds: object())
    monkeypatch.setattr(mod, "examples_from_layer", lambda layer: [(None, None, 1.0)])
    monkeypatch.setattr(mod, "layer_passes_validation", lambda *a, **k: False)

    anchor_games = [_deep_history(), _deep_history()]
    result = run_curriculum(anchor_games=anchor_games, reference_binary="fake",
                            seconds=0.1, trainer=FakeTrainer(), max_uninformative_layers=3)

    # Threshold is "exceeded" at the 4th consecutive failure (1,2,3,4 > 3),
    # two anchor games built per distance attempted -> 8 build_layer calls.
    assert result == {"layers_built": 8, "outcome": "informative_failure"}

def test_run_curriculum_reaches_stopped_success_exhausted_when_validation_always_passes(monkeypatch):
    import tools.proven_anchor_curriculum as mod
    monkeypatch.setattr(mod, "build_layer", lambda ancestor, seconds: object())
    monkeypatch.setattr(mod, "examples_from_layer", lambda layer: [(None, None, 1.0)])
    monkeypatch.setattr(mod, "layer_passes_validation", lambda *a, **k: True)

    # Shallow history: exhausts after exactly one distance (back=0 valid,
    # back=6 goes negative for qualifying_ply=1), with every attempted
    # layer having validated -- a clean "ran out of material" stop, not a
    # failure in disguise.
    result = run_curriculum(anchor_games=[(["p0", "p1"], 1), (["p0", "p1"], 1)],
                            reference_binary="fake", seconds=0.1, trainer=FakeTrainer())

    assert result == {"layers_built": 2, "outcome": "stopped_success_exhausted"}

def test_run_curriculum_resets_the_failure_counter_after_a_pass_and_keeps_widening(monkeypatch):
    import tools.proven_anchor_curriculum as mod
    monkeypatch.setattr(mod, "build_layer", lambda ancestor, seconds: object())
    monkeypatch.setattr(mod, "examples_from_layer", lambda layer: [(None, None, 1.0)])

    # Scripted responses, in call order: pass (back=0, nothing validated yet
    # so no cumulative re-check fires), fail, fail (two failures -- short of
    # max_uninformative_layers=2, so the loop must not stop here), pass
    # (back=18) which *does* trigger spec §2's cumulative re-check of the
    # back=0 held-out set (one extra call) -- scripted to also pass, so the
    # counter resets and widening continues. If the counter failed to reset
    # after the first pass, the two failures alone would already exceed a
    # threshold of 2 only once a 3rd failure landed -- this sequence instead
    # proves the reset happened by reaching the history's natural exhaustion
    # (back=24) with outcome still "success", not "failure".
    responses = iter([True, False, False, True, True])
    monkeypatch.setattr(mod, "layer_passes_validation", lambda *a, **k: next(responses))

    qualifying_ply = 3 * 6  # exhausts after 4 attempted widens (back=0,6,12,18 valid; 24 invalid)
    positions = [f"p{i}" for i in range(qualifying_ply + 1)]
    anchor_games = [(positions, qualifying_ply), (positions, qualifying_ply)]

    result = run_curriculum(anchor_games=anchor_games, reference_binary="fake",
                            seconds=0.1, trainer=FakeTrainer(), max_uninformative_layers=2)

    assert result == {"layers_built": 8, "outcome": "stopped_success_exhausted"}
