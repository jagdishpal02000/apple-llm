"""The dialect rules, and the shared golden corpus.

Runs anywhere -- nothing here needs Apple hardware.
"""

import json

import pytest
from conftest import load_fixtures

from apple_llm import to_apple_schema

FIXTURES = load_fixtures()


def test_corpus_is_not_empty():
    assert len(FIXTURES) > 10


@pytest.mark.parametrize("fixture", FIXTURES, ids=[f["name"] for f in FIXTURES])
def test_golden_corpus(fixture):
    """The same files the Node suite reads. A divergence between the two ports
    shows up here as a mismatch against a shared expectation."""
    assert to_apple_schema(fixture["input"]) == fixture["expected"], fixture["why"]


def test_rule_1_never_collapses_a_multi_type_union():
    # Collapsing ["number","string"] to "string" made it physically impossible
    # for the model to emit a status code; it wrote ": 201" instead.
    out = to_apple_schema({"type": "object", "properties": {"s": {"type": ["number", "string"]}}})
    assert out["properties"]["s"]["anyOf"] == [{"type": "number"}, {"type": "string"}]
    assert "type" not in out["properties"]["s"]


def test_rule_1_nullability_is_absence_from_required():
    out = to_apple_schema({
        "type": "object",
        "required": ["a", "b"],
        "properties": {"a": {"type": "string"}, "b": {"type": ["string", "null"]}},
    })
    assert out["properties"]["b"]["type"] == "string"
    assert out["required"] == ["a"]
    # ...but it is still generated, so it must stay in x-order.
    assert out["x-order"] == ["a", "b"]


def test_rule_2_x_order_covers_every_property():
    out = to_apple_schema({
        "type": "object",
        "properties": {
            "a": {"type": "string"},
            "nested": {"type": "object", "properties": {"z": {"type": "string"}}},
        },
    })
    assert out["x-order"] == ["a", "nested"]
    assert out["properties"]["nested"]["x-order"] == ["z"]


def test_rule_3_no_bare_enum_survives():
    out = to_apple_schema({"type": "object", "properties": {"e": {"enum": ["x", "y"]}}})
    e = out["properties"]["e"]
    assert "enum" not in e
    assert e["anyOf"] == [
        {"type": "string", "const": "x"},
        {"type": "string", "const": "y"},
    ]


def test_rule_4_objects_and_anyof_carry_a_title():
    out = to_apple_schema({"type": "object", "properties": {"e": {"enum": ["x"]}}})
    assert isinstance(out["title"], str)
    assert isinstance(out["properties"]["e"]["title"], str)


def test_rule_4_ref_is_rewritten_when_the_def_name_is_sanitized():
    # $ref resolves by title, so a name that cannot be a title would otherwise
    # resolve to nothing -- silently.
    out = to_apple_schema({
        "type": "object",
        "properties": {"i": {"$ref": "#/$defs/my-param"}},
        "$defs": {"my-param": {"type": "object", "properties": {"n": {"type": "string"}}}},
    })
    assert out["$defs"]["my-param"]["title"] == "myparam"
    assert out["properties"]["i"]["$ref"] == "#/$defs/myparam"


def test_rule_5_every_object_states_additional_properties():
    out = to_apple_schema({"type": "object", "properties": {"a": {"type": "string"}}})
    assert out["additionalProperties"] is False


def test_rule_6_numeric_enum_keeps_its_type():
    # `const` decodes as String only, so 200 cannot be a const. Retyping the
    # field to string would be the silent failure; dropping the literal is not.
    out = to_apple_schema({"type": "object", "properties": {"s": {"enum": [200, 404]}}})
    s = out["properties"]["s"]
    assert s["type"] == "integer"
    assert "anyOf" not in s
    assert '"200"' not in json.dumps(s)


def test_rule_6_boolean_enum_becomes_a_boolean():
    out = to_apple_schema({"type": "object", "properties": {"b": {"enum": [True, False]}}})
    assert out["properties"]["b"]["type"] == "boolean"


def test_rule_7_empty_object_still_gets_properties():
    out = to_apple_schema({"type": "object", "properties": {"m": {"type": "object"}}})
    assert out["properties"]["m"]["properties"] == {}
    assert out["properties"]["m"]["additionalProperties"] is False


def test_caller_supplied_x_order_is_honoured():
    out = to_apple_schema({
        "type": "object",
        "x-order": ["b"],
        "properties": {"a": {"type": "string"}, "b": {"type": "string"}},
    })
    assert out["x-order"] == ["b", "a"]


def test_distinct_objects_get_distinct_titles():
    # Two `items` at different levels would otherwise both be typed `items`.
    out = to_apple_schema({
        "type": "object",
        "properties": {
            "items": {"type": "object", "properties": {"a": {"type": "string"}}},
            "other": {
                "type": "array",
                "items": {"type": "object", "properties": {"b": {"type": "string"}}},
            },
        },
    })
    titles = {out["properties"]["items"]["title"], out["properties"]["other"]["items"]["title"]}
    assert len(titles) == 2


def test_is_idempotent_on_its_own_output():
    once = to_apple_schema({
        "type": "object",
        "required": ["a"],
        "properties": {"a": {"type": "string"}, "b": {"enum": ["x", "y"]}},
    })
    assert to_apple_schema(once) == once
