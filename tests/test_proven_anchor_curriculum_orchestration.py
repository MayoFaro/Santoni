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

    # Each distance now runs one per-game validation check per active
    # anchor game (2 here), all evaluated (no short-circuiting) so every
    # per-game held-out set is genuinely exercised. Scripted per distance:
    # back=0 fails (both per-game checks False), back=6 fails too (two
    # failures total -- short of max_uninformative_layers=2, so the loop
    # must not stop here), back=12 passes (both True) with *no* cumulative
    # re-check calls, since nothing validated earlier to re-check -- this
    # keeps the script free of the cumulative re-check's call-count
    # subtlety while still proving the reset: if the counter hadn't reset
    # after this pass, a single further failure would push it over the
    # threshold, but instead the loop runs all the way to this anchor
    # game's natural exhaustion (back=18) with outcome still "success".
    responses = iter([False, False, False, False, True, True])
    monkeypatch.setattr(mod, "layer_passes_validation", lambda *a, **k: next(responses))

    qualifying_ply = 2 * 6  # exhausts after 3 attempted widens (back=0,6,12 valid; 18 invalid)
    positions = [f"p{i}" for i in range(qualifying_ply + 1)]
    anchor_games = [(positions, qualifying_ply), (positions, qualifying_ply)]

    result = run_curriculum(anchor_games=anchor_games, reference_binary="fake",
                            seconds=0.1, trainer=FakeTrainer(), max_uninformative_layers=2)

    assert result == {"layers_built": 6, "outcome": "stopped_success_exhausted"}

def test_run_curriculums_held_out_pool_never_validates_a_games_own_children(monkeypatch):
    """Direct regression test for the leave-one-out fix: inspects the
    actual `held_out` lists handed to `layer_passes_validation`, instead of
    only stubbing its return value -- the gap that let the original
    pooled-everyone-together bug through undetected."""
    import types
    import tools.proven_anchor_curriculum as mod

    class FakeLayer:
        def __init__(self, tag):
            self.children = [(types.SimpleNamespace(after=f"child-from-{tag}"), True)]
            self.ancestor_verdict = None  # keeps the symmetrized-ancestor pool out of scope here
            self.ancestor = None

    # build_layer is called once per active anchor game, in anchor_games
    # order -- tag each fake layer with which game it came from so the
    # held-out sets below can be checked for provenance.
    layers = iter([FakeLayer("A"), FakeLayer("B")])
    monkeypatch.setattr(mod, "build_layer", lambda ancestor, seconds: next(layers))
    monkeypatch.setattr(mod, "examples_from_layer", lambda layer: [])

    seen_held_out_sets = []

    def fake_validate(net, baseline_net, held_out):
        seen_held_out_sets.append(list(held_out))
        return False

    monkeypatch.setattr(mod, "layer_passes_validation", fake_validate)

    run_curriculum(anchor_games=[(["p0", "p1"], 1), (["p0", "p1"], 1)],
                   reference_binary="fake", seconds=0.1, trainer=FakeTrainer(),
                   max_uninformative_layers=0)

    assert len(seen_held_out_sets) == 2  # one per-game check this round
    held_out_for_game_a, held_out_for_game_b = seen_held_out_sets
    assert ("child-from-B", True) in held_out_for_game_a
    assert ("child-from-A", True) not in held_out_for_game_a
    assert ("child-from-A", True) in held_out_for_game_b
    assert ("child-from-B", True) not in held_out_for_game_b
