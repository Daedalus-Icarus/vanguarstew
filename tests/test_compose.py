"""Tests for the composite score (judge + objective anchor blended into [0, 1])."""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from benchmark.score import composite_score, objective_component  # noqa: E402


def test_objective_component_module_recall_only():
    assert objective_component({"module_recall": 0.5}) == 0.5
    assert objective_component({}) == 0.0


def test_objective_component_counts_release_only_when_signaled():
    # no release in window -> only module recall counts
    assert objective_component({"module_recall": 0.4, "release_signaled": False}) == 0.4
    # release happened and was predicted -> averaged in as 1.0
    assert objective_component({"module_recall": 1.0, "release_signaled": True,
                                "release_predicted": True}) == 1.0
    # release happened but was missed -> averaged in as 0.0
    assert objective_component({"module_recall": 1.0, "release_signaled": True,
                                "release_predicted": False}) == 0.5


def test_objective_component_includes_bump_when_present():
    obj = {"module_recall": 1.0, "release_signaled": True, "release_predicted": True,
           "bump_actual": "minor", "bump_match": True}
    assert objective_component(obj) == 1.0
    obj["bump_match"] = False
    assert objective_component(obj) == round(2 / 3, 3)


def test_objective_component_prefers_weighted_module_recall():
    # When file-weighted recall is present it is used instead of plain recall, so the
    # score reflects where change actually concentrated (#61).
    assert objective_component({"module_recall": 0.5, "weighted_module_recall": 0.9}) == 0.9
    # It blends with the release/bump signals exactly like plain recall does.
    obj = {"module_recall": 0.2, "weighted_module_recall": 0.8,
           "release_signaled": True, "release_predicted": True}
    assert objective_component(obj) == round((0.8 + 1.0) / 2, 3)


def test_objective_component_falls_back_to_plain_recall_when_unweighted():
    # No weighted recall available (e.g. the weighted producer is not present yet):
    # plain module_recall is used, so behavior is unchanged until it lands.
    assert objective_component({"module_recall": 0.5}) == 0.5
    # An explicit None weighted value falls back rather than being treated as 0.0.
    assert objective_component({"module_recall": 0.4, "weighted_module_recall": None}) == 0.4


def test_objective_component_ignores_backlog_recall():
    empty = {"module_recall": 0.5, "backlog_recall": 0.0,
             "addressed_issue_numbers": [], "matched_issue_numbers": []}
    addressed = {"module_recall": 0.5, "backlog_recall": 0.0,
                 "addressed_issue_numbers": [12], "matched_issue_numbers": []}
    matched = {"module_recall": 0.5, "backlog_recall": 1.0,
               "addressed_issue_numbers": [12], "matched_issue_numbers": [12]}
    assert objective_component(empty) == 0.5
    assert objective_component(addressed) == 0.5
    assert objective_component(matched) == 0.5


def test_composite_uses_weighted_recall_end_to_end():
    # The composite reflects weighted recall through objective_component (#61).
    obj = {"module_recall": 0.0, "weighted_module_recall": 1.0}
    assert composite_score("tie", obj) == 0.7  # 0.6*0.5 + 0.4*1.0


def test_composite_ignores_backlog_recall_for_empty_and_addressed_backlogs():
    empty = {"module_recall": 0.5, "backlog_recall": 0.0,
             "addressed_issue_numbers": [], "matched_issue_numbers": []}
    addressed = {"module_recall": 0.5, "backlog_recall": 0.0,
                 "addressed_issue_numbers": [12], "matched_issue_numbers": []}
    matched = {"module_recall": 0.5, "backlog_recall": 1.0,
               "addressed_issue_numbers": [12], "matched_issue_numbers": [12]}
    expected = composite_score("A", empty)
    assert expected == 0.8
    assert composite_score("A", addressed) == expected
    assert composite_score("A", matched) == expected


def test_composite_blends_judge_and_objective():
    obj = {"module_recall": 0.5}
    assert composite_score("A", obj) == 0.8    # 0.6*1.0 + 0.4*0.5
    assert composite_score("B", obj) == 0.2    # 0.6*0.0 + 0.4*0.5
    assert composite_score("tie", obj) == 0.5  # 0.6*0.5 + 0.4*0.5


def test_composite_weights_are_normalized():
    obj = {"module_recall": 1.0}
    # judge-only weighting -> pure judge outcome
    assert composite_score("A", obj, w_judge=1.0, w_objective=0.0) == 1.0
    assert composite_score("B", obj, w_judge=1.0, w_objective=0.0) == 0.0
    # weights that don't sum to 1 are normalized
    assert composite_score("A", obj, w_judge=3.0, w_objective=1.0) == 1.0
