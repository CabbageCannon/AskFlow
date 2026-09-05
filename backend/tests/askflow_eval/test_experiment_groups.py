from experiment_groups import (
    DEFAULT_STATIC_MODEL,
    GROUPS,
    resolve_experiment_group,
)


def test_groups_a_and_b_share_exact_static_model():
    assert GROUPS["A"].static_model == DEFAULT_STATIC_MODEL
    assert GROUPS["B"].static_model == DEFAULT_STATIC_MODEL
    assert GROUPS["A"].static_model == GROUPS["B"].static_model


def test_group_c_is_dynamic_and_has_no_static_model():
    assert GROUPS["C"].router_mode == "dynamic"
    assert GROUPS["C"].static_model is None


def test_group_roles_are_explicit():
    assert GROUPS["A"].agent == "baseline"
    assert GROUPS["B"].agent == "askflow"
    assert GROUPS["C"].agent == "askflow"


def test_static_model_can_be_overridden_for_a_or_b():
    model = "openai:test-model"
    assert resolve_experiment_group("a", static_model=model).static_model == model
    assert resolve_experiment_group("B", static_model=model).static_model == model
    assert resolve_experiment_group("C", static_model=model).static_model is None
