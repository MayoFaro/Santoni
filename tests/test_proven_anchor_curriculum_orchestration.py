import itertools
from dataclasses import dataclass

import pytest

import tools.proven_anchor_curriculum as mod
from tools.proven_anchor_curriculum import run_curriculum
from tools.proven_anchor.validation import ValidationMeasurement


@pytest.fixture(autouse=True)
def stub_engine_pinning(monkeypatch):
    """`run_curriculum` pins the frozen reference engine before anything
    else (so proof search can never silently run against whatever happens
    to be built). These tests pass placeholder binary paths, so the pin is
    stubbed out here -- the real call has its own test below, and the
    end-to-end test re-installs the genuine `set_engine`."""
    monkeypatch.setattr(mod, "set_engine", lambda binary: None)


class FakeNet:
    """Minimal stand-in for `PolicyValueNet`'s train/eval protocol -- the
    orchestrator deep-copies the live net and evaluates the copy."""

    def __init__(self):
        self.training = True

    def eval(self):
        self.training = False
        return self

    def train(self, mode=True):
        self.training = mode
        return self


class FakeTrainer:
    def __init__(self):
        self.examples = []
        self.net = FakeNet()
        self.train_steps = 0

    def add_exact_examples(self, examples):
        self.examples.extend(examples)

    def train_step(self, batch_size=64):
        self.train_steps += 1
        return 0.1


@dataclass(frozen=True)
class FakePosition:
    name: str
    winner: int | None = None


@dataclass(frozen=True)
class FakeTurn:
    after: FakePosition


class FakeLayer:
    """Stands in for a `LayerResult` with a known set of child verdicts."""

    def __init__(self, tag, verdicts=(True, True, False, False), resolved_fraction=1.0,
                 children=None):
        self.children = children if children is not None else [
            (FakeTurn(FakePosition(f"child{i}-from-{tag}")), verdict)
            for i, verdict in enumerate(verdicts)
        ]
        self.resolved_fraction = resolved_fraction
        self.ancestor = None
        self.ancestor_verdict = None  # keeps the symmetrized pool out of scope


def fake_layers(monkeypatch, tags=('A', 'B'), **kwargs):
    """Hand out one `FakeLayer` per `build_layer` call, cycling the tags so
    each distance's first layer is game A's and the second is game B's."""
    tag_cycle = itertools.cycle(tags)
    monkeypatch.setattr(mod, "build_layer",
                        lambda ancestor, seconds: FakeLayer(next(tag_cycle), **kwargs))
    monkeypatch.setattr(mod, "examples_from_layer", lambda layer: [("ancestor-example",)])
    monkeypatch.setattr(mod, "child_examples_from_layer",
                        lambda layer, children=None: [("child-example",)])


def a_measurement(passed):
    return ValidationMeasurement(
        rate=0.9 if passed else 0.4, ci_lo=0.8 if passed else 0.2,
        ci_hi=0.95 if passed else 0.6, baseline_rate=0.5, n=8,
        true_labels=4, false_labels=4, majority_baseline_rate=0.5, passed=passed,
    )


def scripted_validation(monkeypatch, outcomes, seen=None):
    """Install a `measure_layer_validation` that returns `outcomes` in
    order (the last value repeats if the script runs out)."""
    remaining = list(outcomes)

    def fake_measure(net, baseline_net, held_out):
        if seen is not None:
            seen.append(list(held_out))
        passed = remaining.pop(0) if len(remaining) > 1 else remaining[0]
        return a_measurement(passed)

    monkeypatch.setattr(mod, "measure_layer_validation", fake_measure)


def deep_history(qualifying_ply=120, prefix='g'):
    """Long enough that no distance explored by these tests ever runs an
    anchor game out of its own history -- isolates the attempt-counter
    logic from the exhaustion stop condition. Distinct position names per
    game, since two games sharing a position is itself a condition the
    orchestrator now reacts to."""
    return [f"{prefix}{i}" for i in range(qualifying_ply + 1)], qualifying_ply


def test_run_curriculum_requires_at_least_two_anchor_games():
    with pytest.raises(ValueError):
        run_curriculum(anchor_games=[(["p0"], 0)], reference_binary="fake",
                       seconds=0.1, trainer=FakeTrainer())


def test_a_failing_distance_is_retrained_and_retried_never_widened(monkeypatch):
    """Spec Sec 3.4/3.6: the gate is obligatory *before* widening, so a
    failure means more training at the same distance -- advancing to a
    harder distance on failure made the gate structurally unclearable."""
    fake_layers(monkeypatch)
    scripted_validation(monkeypatch, [False])

    trainer = FakeTrainer()
    result = run_curriculum(anchor_games=[deep_history(prefix='a'), deep_history(prefix='b')],
                            reference_binary="fake", seconds=0.1, trainer=trainer,
                            max_uninformative_layers=2, train_steps_per_attempt=3,
                            progress=None)

    assert result["outcome"] == "informative_failure"
    # One distance only: two layers built (one per anchor game), never rebuilt.
    assert result["layers_built"] == 2
    assert {record["distance"] for record in result["rounds"]} == {0}
    assert [record["attempt"] for record in result["rounds"]] == [1, 2, 3]
    # Each attempt spends a real training budget, not a single gradient step.
    assert trainer.train_steps == 3 * 3


def test_a_passing_distance_widens_by_one_step(monkeypatch):
    fake_layers(monkeypatch)
    # distance 0: both per-game checks pass (no cumulative sets yet).
    # distance 6: both fail -> stops immediately (budget of 0 retries).
    scripted_validation(monkeypatch, [True, True, False])

    result = run_curriculum(anchor_games=[deep_history(prefix='a'), deep_history(prefix='b')],
                            reference_binary="fake", seconds=0.1, trainer=FakeTrainer(),
                            max_uninformative_layers=0, train_steps_per_attempt=1,
                            progress=None)

    assert [record["distance"] for record in result["rounds"]] == [0, mod.DISTANCE_STEP]
    assert [record["passed"] for record in result["rounds"]] == [True, False]
    assert result["outcome"] == "informative_failure"
    assert result["layers_built"] == 4


def test_run_curriculum_reaches_stopped_success_exhausted_when_validation_always_passes(monkeypatch):
    fake_layers(monkeypatch)
    scripted_validation(monkeypatch, [True])

    # Shallow history: exhausts after exactly one distance (back=0 valid,
    # back=6 goes negative for qualifying_ply=1), with every attempted
    # layer having validated -- a clean "ran out of material" stop, not a
    # failure in disguise.
    result = run_curriculum(anchor_games=[(["a0", "a1"], 1), (["b0", "b1"], 1)],
                            reference_binary="fake", seconds=0.1, trainer=FakeTrainer(),
                            train_steps_per_attempt=1, progress=None)

    assert result["outcome"] == "stopped_success_exhausted"
    assert result["layers_built"] == 2


def test_held_out_pool_is_leave_one_out_and_labelled_from_the_childs_own_perspective(monkeypatch):
    """Two findings in one place: a game's own children never validate its
    own training (leave-one-out), and a child's label is the *negation* of
    its verdict, because a verdict is about the ancestor's mover while the
    label convention is about the position's own player to move."""
    fake_layers(monkeypatch, verdicts=(True, True, False, False))
    seen = []
    scripted_validation(monkeypatch, [False], seen=seen)

    run_curriculum(anchor_games=[(["a0", "a1"], 1), (["b0", "b1"], 1)],
                   reference_binary="fake", seconds=0.1, trainer=FakeTrainer(),
                   max_uninformative_layers=0, train_steps_per_attempt=1, progress=None)

    assert len(seen) == 2  # one per-game check this round
    held_out_for_a, held_out_for_b = seen
    # Children at odd indices are the validation half; verdicts there are
    # True (child 1) and False (child 3), so labels are False and True.
    assert held_out_for_a == [(FakePosition("child1-from-B"), False),
                              (FakePosition("child3-from-B"), True)]
    assert held_out_for_b == [(FakePosition("child1-from-A"), False),
                              (FakePosition("child3-from-A"), True)]
    assert not any(name.name.endswith("from-A") for name, _ in held_out_for_a)


def test_positions_the_network_trained_on_are_excluded_from_every_held_out_pool(monkeypatch):
    """Plan Review Focus: two near-identical anchor games must not let the
    gate compare a position against itself under another name."""
    shared = FakeTurn(FakePosition("shared-position"))
    layer_a = FakeLayer("A", children=[(shared, True),
                                       (FakeTurn(FakePosition("a1")), True),
                                       (FakeTurn(FakePosition("a2")), False),
                                       (FakeTurn(FakePosition("a3")), False)])
    layer_b = FakeLayer("B", children=[(FakeTurn(FakePosition("b0")), True),
                                       (shared, True),  # same position, other game
                                       (FakeTurn(FakePosition("b2")), False),
                                       (FakeTurn(FakePosition("b3")), False)])
    layers = iter([layer_a, layer_b])
    monkeypatch.setattr(mod, "build_layer", lambda ancestor, seconds: next(layers))
    monkeypatch.setattr(mod, "examples_from_layer", lambda layer: [])
    monkeypatch.setattr(mod, "child_examples_from_layer", lambda layer, children=None: [])

    seen = []
    scripted_validation(monkeypatch, [False], seen=seen)

    run_curriculum(anchor_games=[(["a0", "a1"], 1), (["b0", "b1"], 1)],
                   reference_binary="fake", seconds=0.1, trainer=FakeTrainer(),
                   max_uninformative_layers=0, train_steps_per_attempt=1, progress=None)

    held_out_for_a = seen[0]
    # `shared-position` is in game B's validation half, but game A trained
    # on it (its own training half) -- so it cannot be held-out evidence.
    assert [position.name for position, _ in held_out_for_a] == ["b3"]


def test_a_regression_on_an_earlier_distance_is_reported_distinctly(monkeypatch):
    """Spec Sec 2 adds the cumulative re-check so catastrophic forgetting
    does not go unnoticed; folding it into the same bucket as a fresh
    distance failing to clear the bar would defeat its purpose."""
    fake_layers(monkeypatch)
    # distance 0: both per-game checks pass. distance 6: both per-game
    # checks pass, then the cumulative re-check of distance 0 fails.
    scripted_validation(monkeypatch, [True, True, True, True, False])

    result = run_curriculum(anchor_games=[deep_history(prefix='a'), deep_history(prefix='b')],
                            reference_binary="fake", seconds=0.1, trainer=FakeTrainer(),
                            max_uninformative_layers=0, train_steps_per_attempt=1,
                            progress=None)

    assert result["outcome"] == "stopped_regression"
    assert result["regression_at_distance"] == 0
    assert result["rounds"][-1]["regression_at_distance"] == 0
    assert result["rounds"][-1]["passed"] is False


def test_running_out_of_held_out_evidence_is_its_own_clean_outcome(monkeypatch):
    """Expected terminal condition (spec Sec 3.2's natural depth ceiling),
    not an opaque "error" that loses even the layer count. The real
    validation function runs here -- nothing stubs it."""
    monkeypatch.setattr(
        mod, "build_layer",
        lambda ancestor, seconds: FakeLayer("X", verdicts=(None, None, None),
                                            resolved_fraction=0.0))
    monkeypatch.setattr(mod, "examples_from_layer", lambda layer: [])
    monkeypatch.setattr(mod, "child_examples_from_layer", lambda layer, children=None: [])

    result = run_curriculum(anchor_games=[(["a0", "a1"], 1), (["b0", "b1"], 1)],
                            reference_binary="fake", seconds=0.1, trainer=FakeTrainer(),
                            train_steps_per_attempt=1, progress=None)

    assert result["outcome"] == "stopped_unresolvable"
    assert result["layers_built"] == 2  # not lost to a blanket error branch
    assert "no held-out evidence" in result["detail"]


def test_an_unexpected_failure_still_preserves_what_was_measured(monkeypatch):
    calls = itertools.count()

    def explode_on_the_second_distance(ancestor, seconds):
        if next(calls) >= 2:
            raise RuntimeError("reference binary vanished")
        return FakeLayer("A")

    monkeypatch.setattr(mod, "build_layer", explode_on_the_second_distance)
    monkeypatch.setattr(mod, "examples_from_layer", lambda layer: [])
    monkeypatch.setattr(mod, "child_examples_from_layer", lambda layer, children=None: [])
    scripted_validation(monkeypatch, [True])

    result = run_curriculum(anchor_games=[deep_history(prefix='a'), deep_history(prefix='b')],
                            reference_binary="fake", seconds=0.1, trainer=FakeTrainer(),
                            train_steps_per_attempt=1, progress=None)

    assert result["outcome"] == "error"
    assert result["layers_built"] == 2
    assert len(result["rounds"]) == 1


def test_resolved_fractions_and_validation_numbers_reach_the_summary_and_the_output(monkeypatch):
    """Spec Sec 3.2's resolved fraction and Sec 2's rate/CI/baseline are
    the run's actual deliverables -- computing them and dropping them on
    the floor is not reporting them."""
    fake_layers(monkeypatch, resolved_fraction=0.75)
    scripted_validation(monkeypatch, [False])

    printed = []
    result = run_curriculum(anchor_games=[(["a0", "a1"], 1), (["b0", "b1"], 1)],
                            reference_binary="fake", seconds=0.1, trainer=FakeTrainer(),
                            max_uninformative_layers=0, train_steps_per_attempt=1,
                            progress=printed.append)

    assert result["resolved_fractions"] == [{"distance": 0, "per_game": [0.75, 0.75]}]
    record = result["rounds"][0]
    assert record["resolved_fractions"] == [0.75, 0.75]
    assert record["games"][0]["rate"] == pytest.approx(0.4)
    assert record["games"][0]["ci_lo"] == pytest.approx(0.2)
    assert record["games"][0]["baseline_rate"] == pytest.approx(0.5)
    assert record["games"][0]["majority_baseline_rate"] == pytest.approx(0.5)
    assert record["games"][0]["n"] == 8

    assert len(printed) == 1
    line = printed[0]
    for fragment in ("distance=0", "attempt=1", "resolved=0.75,0.75", "rate=0.400",
                     "CI95=[0.200,0.600]", "base=0.500", "maj=0.500", "n=8", "FAIL"):
        assert fragment in line


def test_validation_never_touches_the_live_training_network(monkeypatch):
    """BatchNorm running statistics are updated by any train-mode forward
    pass, `torch.no_grad()` included -- so validating `trainer.net` itself
    would silently rewrite the network being trained (and every checkpoint
    saved from it) as a side effect of merely checking it."""
    fake_layers(monkeypatch)
    trainer = FakeTrainer()
    inspected = []

    def fake_measure(net, baseline_net, held_out):
        inspected.append((net, baseline_net))
        return a_measurement(False)

    monkeypatch.setattr(mod, "measure_layer_validation", fake_measure)

    run_curriculum(anchor_games=[(["a0", "a1"], 1), (["b0", "b1"], 1)],
                   reference_binary="fake", seconds=0.1, trainer=trainer,
                   max_uninformative_layers=0, train_steps_per_attempt=1, progress=None)

    assert inspected
    for net, baseline_net in inspected:
        assert net is not trainer.net
        assert baseline_net is not trainer.net
        assert net.training is False
        assert baseline_net.training is False
    # The live network is left exactly as training left it.
    assert trainer.net.training is True


def test_run_curriculum_pins_the_frozen_reference_engine(monkeypatch):
    """`reference_binary` must actually pin the engine proof search runs
    against, on every call path -- not just incidentally, via an earlier
    `find_anchor_games` having mutated the process-global library."""
    order = []
    monkeypatch.setattr(mod, "set_engine", lambda binary: order.append(("set_engine", binary)))
    monkeypatch.setattr(mod, "build_layer",
                        lambda ancestor, seconds: order.append(("build_layer",)) or FakeLayer("A"))
    monkeypatch.setattr(mod, "examples_from_layer", lambda layer: [])
    monkeypatch.setattr(mod, "child_examples_from_layer", lambda layer, children=None: [])
    scripted_validation(monkeypatch, [False])

    run_curriculum(anchor_games=[(["a0", "a1"], 1), (["b0", "b1"], 1)],
                   reference_binary="/frozen/libsantoni_engine.so", seconds=0.1,
                   trainer=FakeTrainer(), max_uninformative_layers=0,
                   train_steps_per_attempt=1, progress=None)

    assert order[0] == ("set_engine", "/frozen/libsantoni_engine.so")
    assert ("build_layer",) in order


# --- Real layers, real validation: no stub between the two label conventions ---

def real_ancestors():
    from santorini.engine import Position
    a = Position(workers=((0, 4), (20, 24)), player=0)
    b = Position(workers=((0, 4), (20, 24)), heights=(0, 1, 0, 1) + (0,) * 21, player=0)
    a.validate()
    b.validate()
    return a, b


def cycling_verdicts():
    """Proof-free but realistically shaped verdicts: a quarter of the
    children proven winning for the ancestor's mover, half proven losing,
    a quarter left unresolved (so `resolved_fraction` is genuinely < 1)."""
    counter = itertools.count()

    def evaluate(child, mover, seconds):
        return (True, False, False, None)[next(counter) % 4]

    return evaluate


def test_both_held_out_pools_label_a_position_from_its_own_movers_perspective():
    """The one convention both pools must obey, checked on a real
    `LayerResult` rather than a stub: a label is True exactly when the
    labelled position's *own* player to move is the proven winner. A
    child's verdict is about the ancestor's mover instead, so it has to be
    flipped; a board symmetry preserves `player`, so the ancestor's own
    verdict passes through. Concatenating the two pools under two
    different conventions is exactly the bug this pins down."""
    from tools.proven_anchor.layer import build_layer

    ancestor, _ = real_ancestors()
    layer = build_layer(ancestor, seconds=0.0, evaluate=cycling_verdicts())
    assert layer.ancestor_verdict is True
    assert 0.0 < layer.resolved_fraction < 1.0

    _for_training, held_out_children = mod._partition_children(layer)
    children_pool = mod._children_held_out(held_out_children)
    symmetrized_pool = mod._symmetrized_ancestor_held_out(layer)
    assert children_pool and symmetrized_pool

    verdict_of = {turn.after: verdict for turn, verdict in held_out_children}
    opponent = 1 - ancestor.player
    for position, label in children_pool:
        assert position.player == opponent  # the engine flips the side to move
        proven_winner = ancestor.player if verdict_of[position] else opponent
        assert label == (position.player == proven_winner)
    for position, label in symmetrized_pool:
        assert position.player == ancestor.player  # symmetries preserve it
        assert label == (position.player == ancestor.player)

    # Stated the other way round, to catch an "both pools wrong the same
    # way" fix: a child proven *winning for the ancestor's mover* is a
    # position whose own mover is losing, so its label must be False.
    assert any(verdict is True for verdict in verdict_of.values())
    for position, label in children_pool:
        if verdict_of[position] is True:
            assert label is False


def test_run_curriculum_end_to_end_with_the_real_validation_gate(tmp_path, monkeypatch):
    """One pass through the real plumbing -- real positions, real planes,
    real `Trainer`, real `measure_layer_validation` -- because every other
    orchestration test here stubs the gate, which is precisely how two
    sign/eval-mode bugs survived eight task reviews.

    Only `build_layer`'s per-child proof search is replaced (by fixed
    verdicts), so the test needs no 30-second engine budget; everything
    downstream of it runs for real."""
    import torch
    from santoni_az.trainer import Trainer
    from santorini.native import library_path
    from tools.mcts_az_vs_reference_campaign import set_engine as real_set_engine
    from tools.proven_anchor.layer import build_layer as real_build_layer

    monkeypatch.setattr(mod, "set_engine", real_set_engine)
    evaluate = cycling_verdicts()
    monkeypatch.setattr(mod, "build_layer",
                        lambda ancestor, seconds: real_build_layer(ancestor, seconds,
                                                                   evaluate=evaluate))

    ancestor_a, ancestor_b = real_ancestors()
    games_dir = tmp_path / 'games'
    checkpoint_dir = tmp_path / 'checkpoints'
    games_dir.mkdir()
    checkpoint_dir.mkdir()
    trainer = Trainer(games_dir=games_dir, checkpoint_dir=checkpoint_dir)
    before = {name: buffer.clone() for name, buffer in trainer.net.named_buffers()}

    printed = []
    result = run_curriculum(anchor_games=[([ancestor_a], 0), ([ancestor_b], 0)],
                            reference_binary=str(library_path()), seconds=0.0,
                            trainer=trainer, max_uninformative_layers=0,
                            train_steps_per_attempt=2, progress=printed.append)

    assert result["layers_built"] == 2
    # Both outcomes are legitimate here (the gate may or may not clear a
    # baseline after two gradient steps); what must hold is that the run
    # completed and reported real numbers either way.
    assert result["outcome"] in {"informative_failure", "stopped_success_exhausted"}
    assert "error" not in result

    fractions = result["resolved_fractions"]
    assert len(fractions) == 1 and fractions[0]["distance"] == 0
    assert all(0.0 < fraction < 1.0 for fraction in fractions[0]["per_game"])

    record = result["rounds"][0]
    assert len(record["games"]) == 2
    for game in record["games"]:
        assert game["n"] > 0
        assert game["true_labels"] + game["false_labels"] == game["n"]
        assert 0.0 <= game["ci_lo"] <= game["rate"] <= game["ci_hi"] <= 1.0
        assert 0.0 <= game["baseline_rate"] <= 1.0
        # Both label classes must be present, otherwise a constant-output
        # network would clear the gate for free.
        assert game["true_labels"] > 0 and game["false_labels"] > 0
    assert printed and "distance=0" in printed[0]

    # Training (not validation) is the only thing allowed to move the live
    # network's BatchNorm statistics, and it leaves it in train mode.
    assert trainer.net.training is True
    assert any(not torch.equal(before[name], buffer)
               for name, buffer in trainer.net.named_buffers())
