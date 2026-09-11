"""Translate standard JSON Schema into Apple's ``GenerationSchema`` dialect.

Apple's ``GenerationSchema`` is ``Decodable`` from JSON Schema, but only accepts a
restricted dialect. These rules were established empirically -- rules 1-5 in
api-scribe, by decoding a real schema against macOS 26 and 27 until it was
accepted; rules 6 and 7 here, by decoding this package's fixture corpus against
macOS 27 (api-scribe hit neither, having only string enums and no empty objects):

 1. Union types are rejected -- ``type: ['string','null']`` fails with "Expected
    value of type String". Nullability is expressed by leaving the property out
    of ``required`` instead.
 2. Every object schema requires an ``x-order`` array naming its properties in
    generation order. Omitting it fails with "Key 'x-order' not found".
 3. ``enum`` is not recognized. A node may only carry ``type``, ``const``,
    ``$ref`` or ``anyOf``, so enums become ``anyOf`` of ``const`` branches.
 4. Every object schema *and* every ``anyOf`` schema requires a ``title``, and
    ``$ref`` resolves by that title rather than by JSON pointer -- so a
    definition's title must equal the key it is referenced by.
 5. Every object schema must state ``additionalProperties``; omitting it fails
    with "Key 'additionalProperties' not found". macOS 26's decoder tolerated its
    absence, so this one only surfaced on macOS 27.
 6. ``const`` is decoded as a String, always. A numeric member (``const: 200``)
    fails with "Expected value of type String". Stringifying it is accepted but
    changes the output type -- the model then emits ``"200"`` rather than ``200``
    -- so a numeric enum is converted to its underlying type instead and the
    literal constraint is dropped.
 7. An object with no properties still needs an explicit ``properties: {}``.
    Without it Apple reads ``additionalProperties`` in its other JSON Schema
    sense -- a schema for the values -- and fails with "Expected value of type
    Dictionary<String, Any>".
 8. A ``title`` on a ``type: "string"`` node is rejected: Apple reads it as a
    "named string type" and fails with "Named string types must have a
    non-empty enum field". Other primitives (integer, number, boolean) and
    arrays accept a title happily. Titles are therefore stripped from plain
    strings. This matters most for pydantic, whose ``model_json_schema()``
    titles every single property.

``$ref`` / ``$defs`` are otherwise supported and pass through untouched.

Why this matters more than it looks: constrained decoding makes a schema mistake
invisible but total. Collapsing a ``["number","string"]`` union to ``"string"``
made it physically impossible for the model to emit a status code -- it wrote
``": 201"`` and the literal ``"default"`` instead. Never collapse a multi-type
union to one branch; convert it to ``anyOf``.

This is a port of ``packages/node/src/schema.ts`` and must agree with it exactly;
both are tested against the shared corpus in ``tests/fixtures/schema/``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

JsonSchema = dict[str, Any]

#: Keys whose value is a map of name -> schema.
_SCHEMA_MAP_KEYS = ("properties", "$defs", "definitions")
#: Keys whose value is an array of schemas.
_SCHEMA_LIST_KEYS = ("anyOf", "oneOf", "allOf")
#: Keys whose value is a single schema.
_SCHEMA_KEYS = ("items", "not", "additionalItems")

_NOT_NAME = re.compile(r"[^A-Za-z0-9_]")


@dataclass
class _Context:
    counter: int = 0
    #: Definition key -> the title that definition will carry.
    def_titles: dict[str, str] = field(default_factory=dict)
    #: Every title handed out so far, so generated ones stay unique.
    used: set[str] = field(default_factory=set)


def _is_plain_object(value: Any) -> bool:
    return isinstance(value, dict)


def _sanitize(name: str, ctx: _Context) -> str:
    """Apple uses ``title`` as the generated type name *and* resolves ``$ref``
    against it rather than by JSON pointer, so a definition's title has to match
    the key it is referenced by. That rules out any prettifying of the name."""
    cleaned = _NOT_NAME.sub("", name)
    if not cleaned or cleaned[0].isdigit():
        ctx.counter += 1
        return f"Schema{ctx.counter}"
    return cleaned


def _title_from(name: str, ctx: _Context) -> str:
    """A title that is not already taken.

    Titles are Apple's generated type names, so two structurally different
    objects sharing one is a real collision and not just untidy -- nested objects
    take their title from the property that holds them, and a schema with
    ``items`` at two different levels would otherwise produce two different types
    both called ``items``.
    """
    base = _sanitize(name, ctx)
    if base not in ctx.used:
        ctx.used.add(base)
        return base
    i = 2
    while True:
        candidate = f"{base}{i}"
        if candidate not in ctx.used:
            ctx.used.add(candidate)
            return candidate
        i += 1


def _member_type(value: Any) -> str:
    """JSON Schema type name for an enum member."""
    # bool is a subclass of int in Python, so it has to be checked first.
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    return "string"


def _rewrite_ref(ref: str, ctx: _Context) -> str:
    slash = ref.rfind("/")
    if slash == -1:
        return ref
    name = ref[slash + 1 :]
    title = ctx.def_titles.get(name)
    if title is None or title == name:
        return ref
    return f"{ref[: slash + 1]}{title}"


def _convert_node(node: JsonSchema, name: str, ctx: _Context) -> tuple[JsonSchema, bool]:
    out: JsonSchema = {}
    nullable = False

    # Convert child properties first: a nullable child decides the parent's
    # `required`.
    converted: dict[str, tuple[JsonSchema, bool]] = {}
    properties = node.get("properties")
    if _is_plain_object(properties):
        for key, value in properties.items():
            converted[key] = (
                _convert_node(value, key, ctx) if _is_plain_object(value) else (value, False)
            )

    for key, value in node.items():
        if key in ("type", "enum", "required", "properties"):
            continue

        if key == "$ref" and isinstance(value, str):
            out["$ref"] = _rewrite_ref(value, ctx)
        elif key in _SCHEMA_MAP_KEYS and _is_plain_object(value):
            is_defs = key in ("$defs", "definitions")
            mapped: JsonSchema = {}
            for child_name, child_value in value.items():
                if not _is_plain_object(child_value):
                    mapped[child_name] = child_value
                    continue
                child, _ = _convert_node(child_value, child_name, ctx)
                # Definitions take the title reserved for them in the pre-pass,
                # so the rewritten `$ref`s point at a title that exists.
                if is_defs and child_name in ctx.def_titles:
                    child["title"] = ctx.def_titles[child_name]
                mapped[child_name] = child
            out[key] = mapped
        elif key in _SCHEMA_LIST_KEYS and isinstance(value, list):
            # Apple only understands `anyOf`; `oneOf` is the same shape here.
            # `allOf` is different (must match all, not one) and Apple rejects
            # the key outright. A single-element `allOf` -- the common pydantic
            # pattern `{description, allOf: [{$ref}]}` -- unwraps to that entry;
            # multi-element `allOf` is left to fail loudly as SchemaRejectedError
            # rather than silently mistranslated to `anyOf`.
            if key == "allOf":
                converted_allof = [
                    _convert_node(entry, name, ctx)[0] if _is_plain_object(entry) else entry
                    for entry in value
                ]
                if len(converted_allof) == 1 and _is_plain_object(converted_allof[0]):
                    for k, v in converted_allof[0].items():
                        if k not in out:
                            out[k] = v
                else:
                    out["allOf"] = converted_allof
            else:
                target = "anyOf" if key == "oneOf" else key
                out[target] = [
                    _convert_node(entry, name, ctx)[0] if _is_plain_object(entry) else entry
                    for entry in value
                ]
        elif key in _SCHEMA_KEYS and _is_plain_object(value):
            out[key], _ = _convert_node(value, name, ctx)
        else:
            out[key] = value

    # Rule 1: split union types. `null` becomes optionality; a genuine
    # multi-type union becomes `anyOf`, which Apple does understand. Collapsing
    # such a union to a single type would make the other branches unreachable
    # under constrained decoding -- the model then cannot emit them at all.
    raw_type = node.get("type")
    if isinstance(raw_type, list):
        non_null = [t for t in raw_type if t != "null"]
        nullable = len(non_null) != len(raw_type)
        if len(non_null) > 1:
            out["anyOf"] = [{"type": t} for t in non_null]
        else:
            out["type"] = non_null[0] if non_null else "string"
    elif raw_type is not None:
        out["type"] = raw_type

    # Rule 3: enums become anyOf/const branches. A `null` member is dropped and
    # recorded as nullability, matching rule 1.
    raw_enum = node.get("enum")
    if isinstance(raw_enum, list) and raw_enum:
        members = [v for v in raw_enum if v is not None]
        if len(members) != len(raw_enum):
            nullable = True
        if members:
            out.pop("type", None)
            if all(isinstance(v, str) for v in members):
                out["anyOf"] = [{"type": "string", "const": v} for v in members]
            else:
                # Rule 6: `const` must be a String, so a non-string enum cannot
                # be expressed as const branches. Keep the type and drop the
                # literal constraint rather than stringify -- a silently retyped
                # field is the worse failure, and under constrained decoding the
                # caller cannot see it happen.
                types: list[str] = []
                for v in members:
                    t = _member_type(v)
                    if t not in types:
                        types.append(t)
                if len(types) == 1:
                    out["type"] = types[0]
                else:
                    out["anyOf"] = [{"type": t} for t in types]

    is_object = out.get("type") == "object" or bool(converted)
    if is_object:
        props: JsonSchema = {schema_key: child for schema_key, (child, _) in converted.items()}
        # Rule 7: `properties` must be present even when empty, or Apple reads
        # `additionalProperties` as a value schema and rejects the boolean.
        out["properties"] = props

        original_required = node.get("required") if isinstance(node.get("required"), list) else []
        # Rule 1 (cont.): a nullable property is modelled as simply not required.
        out["required"] = [k for k in original_required if not converted.get(k, (None, False))[1]]
        # Rule 2: object property order must be declared. A caller-supplied
        # x-order is honoured -- it is the generation order, so a schema whose
        # later fields depend on earlier ones has a real reason to choose it --
        # but any property missing from it is appended, since Apple requires
        # every one to be listed.
        keys = list(props.keys())
        declared = node.get("x-order")
        declared_order = (
            [k for k in declared if isinstance(k, str) and k in keys]
            if isinstance(declared, list)
            else []
        )
        out["x-order"] = declared_order + [k for k in keys if k not in declared_order]
        # Rule 4: objects need a title.
        if isinstance(out.get("title"), str):
            ctx.used.add(out["title"])
        else:
            out["title"] = _title_from(name, ctx)
        # Rule 5: every object schema must state `additionalProperties`.
        if "additionalProperties" not in out:
            out["additionalProperties"] = False
    elif isinstance(node.get("required"), list):
        out["required"] = node["required"]

    # Rule 4 (cont.): anyOf/allOf schemas need a title too. Apple only
    # understands `anyOf`, but `allOf` passes through with the same title
    # requirement (different composition semantics, so the key is kept).
    if (
        (isinstance(out.get("anyOf"), list) or isinstance(out.get("allOf"), list))
        and not isinstance(out.get("title"), str)
    ):
        out["title"] = _title_from(name, ctx)

    # Rule 8: a titled string is a "named string type" and must carry an enum.
    # Strip the title instead -- it is only a generated type name, and a plain
    # string does not need one. (A `$defs` entry that is a bare string would
    # lose its `$ref` target here, but such a schema cannot be expressed on
    # Apple at all: a named string type has to be an enum.)
    if out.get("type") == "string" and "anyOf" not in out and "allOf" not in out:
        out.pop("title", None)

    return out, nullable


def _reserve_def_titles(schema: JsonSchema, ctx: _Context) -> None:
    """Reserve each definition's title before converting anything, using the
    same counter the conversion will use. A throwaway counter would let a
    generated ``SchemaN`` name drift between the two passes."""
    for key in ("$defs", "definitions"):
        defs = schema.get(key)
        if not _is_plain_object(defs):
            continue
        for name, value in defs.items():
            declared = value.get("title") if _is_plain_object(value) else None
            title = declared if isinstance(declared, str) else _sanitize(name, ctx)
            ctx.def_titles[name] = title
            ctx.used.add(title)


def to_apple_schema(schema: JsonSchema, root_name: str = "Response") -> JsonSchema:
    """Translate a standard JSON Schema into the dialect Apple's
    ``GenerationSchema`` accepts. Constrained decoding then makes the shape of
    the reply a guarantee rather than a request."""
    ctx = _Context()
    _reserve_def_titles(schema, ctx)
    return _convert_node(schema, root_name, ctx)[0]
