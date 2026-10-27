"""Provider-facing strict JSON Schema for geometry proposals.

Mirrors the local :mod:`geometry` contract in the subset the documented
Responses API structured-output mechanism supports (``text.format`` with
``type: "json_schema"``, strict decoding). Verified against the official
structured-output documentation:

- Root is a plain object; every object sets ``additionalProperties: false``.
- Every object's ``required`` array lists every key in its ``properties``.
- No ``$ref``, ``anyOf``, ``oneOf``, ``allOf``, or ``not`` anywhere; no
  top-level ``enum``. Nested string enums are used for closed vocabularies.
- Nullable numbers use ``type: ["number", "null"]``; unknown depth uses
  ``type: ["string", "null"]`` and local validation enforces its value.

Deliberate divergences from the local contract (local validation stays the
final authority and still enforces everything below):

- ``evidence`` is omitted: the model cannot know our asset hashes, and local
  code attaches evidence after parsing.
- Numeric ranges, hole containment/overlap, confidence/uncertainty coherence,
  and confirmation consistency are NOT schema-expressible; the local
  validator enforces them on every parsed response.
- ``minimum``/``maximum`` on confidence follow the OpenAI-compatible
  convention the documentation is modeled on; if the endpoint rejects them,
  drop the bounds here (local validation is unaffected).

This module only describes the contract. Nothing here transmits anything.
"""

from __future__ import annotations

from typing import Any, Mapping


def _measurement(description: str) -> dict:
    return {
        "type": "object",
        "description": description,
        "additionalProperties": False,
        "required": ["value", "confidence", "uncertainty_mm", "source",
                     "requires_confirmation"],
        "properties": {
            "value": {"type": ["number", "null"]},
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "uncertainty_mm": {"type": ["number", "null"]},
            "source": {"type": "string",
                       "enum": ["observed_pixel", "scale_estimate",
                                "model_inferred", "user_confirmed"]},
            "requires_confirmation": {"type": "boolean"},
        },
    }


def _hole() -> dict:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["x", "y", "diameter", "depth"],
        "properties": {
            "x": _measurement("hole centre x in millimetres from the plate corner"),
            "y": _measurement("hole centre y in millimetres from the plate corner"),
            "diameter": _measurement("hole diameter in millimetres"),
            "depth": {"type": ["string", "null"]},
        },
    }


GEOMETRY_PROPOSAL_JSON_SCHEMA: dict = {
    "type": "object",
    "additionalProperties": False,
    "required": ["classification", "units", "conventions", "length", "width",
                 "thickness", "holes", "scale", "assumptions", "unknowns",
                 "requires_confirmation", "perspective_warning"],
    "properties": {
        "classification": {"type": "string", "enum": ["rectangular_plate"]},
        "units": {"type": "string", "enum": ["mm"]},
        "conventions": {
            "type": "object",
            "additionalProperties": False,
            "required": ["origin", "axes"],
            "properties": {
                "origin": {"type": "string", "enum": ["plate_corner"]},
                "axes": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["x", "y", "z"],
                    "properties": {
                        "x": {"type": "string", "enum": ["length"]},
                        "y": {"type": "string", "enum": ["width"]},
                        "z": {"type": "string", "enum": ["thickness"]},
                    },
                },
            },
        },
        "length": _measurement("plate length in millimetres (x extent)"),
        "width": _measurement("plate width in millimetres (y extent)"),
        "thickness": _measurement("plate thickness in millimetres; null when not visible"),
        "holes": {"type": "array", "minItems": 1, "maxItems": 25, "items": _hole()},
        "scale": {
            "type": "object",
            "additionalProperties": False,
            "required": ["method", "detail"],
            "properties": {
                "method": {"type": "string",
                           "enum": ["user_measured_reference", "ruler_in_image",
                                    "assumed", "unknown"]},
                "detail": {"type": "string"},
            },
        },
        "assumptions": {"type": "array", "maxItems": 20,
                        "items": {"type": "string"}},
        "unknowns": {"type": "array", "maxItems": 20,
                     "items": {"type": "string"}},
        "requires_confirmation": {"type": "array", "maxItems": 60,
                                  "items": {"type": "string"}},
        "perspective_warning": {"type": "string"},
    },
}

#: Structured-output descriptor for the Responses API ``text.format`` field.
#: ``type``/``strict`` follow the documented contract; ``name`` follows the
#: OpenAI-compatible convention and must be confirmed against the Responses
#: reference before first use (harmless either way: servers ignore unknown
#: descriptor metadata or reject loudly, never silently).
RESPONSE_FORMAT_DESCRIPTOR: dict = {
    "type": "json_schema",
    "name": "geometry_proposal",
    "strict": True,
    "schema": GEOMETRY_PROPOSAL_JSON_SCHEMA,
}


def check_strict_compliant(schema: Any) -> list[str]:
    """Return a list of strict-subset violations (empty means compliant).

    Checks, recursively: every object node declares ``additionalProperties:
    False``; every object's ``required`` covers exactly its ``properties``
    keys; no ``$ref``/``anyOf``/``oneOf``/``allOf``/``not`` keys; no
    top-level ``enum``. Used by tests to pin the provider schema's shape.
    """
    violations: list[str] = []

    def visit(node: Any, path: str, is_root: bool) -> None:
        if not isinstance(node, dict):
            return
        for forbidden in ("$ref", "anyOf", "oneOf", "allOf", "not"):
            if forbidden in node:
                violations.append(f"{path}: forbidden key {forbidden!r}")
        if is_root and "enum" in node:
            violations.append(f"{path}: top-level enum is not allowed")
        if node.get("type") == "object" or "properties" in node:
            if node.get("additionalProperties") is not False:
                violations.append(f"{path}: object must set additionalProperties: false")
            properties = node.get("properties", {})
            required = node.get("required", None)
            if not isinstance(required, list) or set(required) != set(properties):
                violations.append(
                    f"{path}: required must list exactly the properties keys")
            for key, child in properties.items():
                visit(child, f"{path}.{key}", False)
        if node.get("type") == "array" and "items" in node:
            visit(node["items"], path + "[]", False)

    visit(schema, "<root>", True)
    return violations
