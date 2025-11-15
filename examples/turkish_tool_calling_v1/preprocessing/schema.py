from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Literal, Optional, Union, get_args
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    field_validator,
    model_validator,
)

NAME_PATTERN = re.compile(r"^[a-z0-9_]{1,64}$")
PROPERTY_PATTERN = re.compile(r"^[a-z0-9_]+$")
SchemaType = Literal["string", "number", "integer", "boolean", "array", "object"]
VALID_SCHEMA_TYPES = set(get_args(SchemaType))


class OpenAIBaseModel(BaseModel):
    """Base configuration shared by OpenAI tool schema models."""

    model_config = ConfigDict(
        extra="forbid", populate_by_name=True, str_strip_whitespace=True
    )


class ToolSchema(OpenAIBaseModel):
    """Recursive JSON Schema node used to describe tool parameters."""

    type: SchemaType
    description: Optional[str] = None
    default: Optional[Any] = None
    enum: Optional[List[Any]] = None
    items: Optional["ToolSchema"] = None
    properties: Dict[str, "ToolSchema"] = Field(default_factory=dict)
    required: List[str] = Field(default_factory=list)
    additional_properties: Optional[Union[bool, "ToolSchema"]] = Field(
        default=None, alias="additionalProperties"
    )

    @field_validator("type", mode="before")
    @classmethod
    def _normalize_type(cls, value: Any) -> SchemaType:
        alias_map = {
            "str": "string",
            "float": "number",
            "bool": "boolean",
            "list": "array",
            "dict": "object",
            "int": "integer",
            "set": "array",
            "tuple": "array",
        }
        if isinstance(value, str):
            lowered = value.lower().strip()
            tokens = [
                token
                for chunk in re.split(r"[\s,|/]+", lowered)
                for token in re.findall(r"[a-z]+", chunk)
            ]
            for token in tokens:
                canonical = alias_map.get(token, token)
                if canonical in VALID_SCHEMA_TYPES:
                    return canonical  # type: ignore[return-value]
        return value  # Pydantic enforces SchemaType after normalization

    @field_validator("enum")
    @classmethod
    def _check_enum(cls, values: Optional[List[Any]]) -> Optional[List[Any]]:
        if isinstance(values, list) and not values:
            raise ValueError("Enum must include at least one value.")
        return values

    @field_validator("properties")
    @classmethod
    def _check_properties(
        cls, properties: Dict[str, "ToolSchema"], info: ValidationInfo
    ) -> Dict[str, "ToolSchema"]:
        if not properties:
            return {}

        cls._ensure_object_schema(info, "properties")
        invalid = [name for name in properties if not PROPERTY_PATTERN.match(name)]
        if invalid:
            raise ValueError(
                "Property names must use snake_case alphanumeric characters: "
                + ", ".join(invalid)
            )
        return properties

    @field_validator("required")
    @classmethod
    def _check_required(cls, required: List[str], info: ValidationInfo) -> List[str]:
        if not required:
            return []

        cls._ensure_object_schema(info, "required")
        defined_props = info.data.get("properties", {})
        missing = [name for name in required if name not in defined_props]
        if missing:
            raise ValueError(
                f"Required keys lack property definitions: {', '.join(missing)}"
            )
        return required

    @field_validator("additional_properties")
    @classmethod
    def _check_additional_properties(
        cls, value: Optional[Union[bool, "ToolSchema"]], info: ValidationInfo
    ) -> Optional[Union[bool, "ToolSchema"]]:
        if value is None:
            return None

        cls._ensure_object_schema(info, "additionalProperties")
        return value

    @model_validator(mode="after")
    def _check_array_items(self) -> "ToolSchema":
        if self.type == "array" and self.items is None:
            self.items = ToolSchema(type="string")
        if self.type != "array" and self.items is not None:
            raise ValueError("'items' can only be defined for array schemas.")
        return self

    @staticmethod
    def _ensure_object_schema(info: ValidationInfo, field_name: str) -> None:
        if info.data.get("type") != "object":
            raise ValueError(f"'{field_name}' can only be defined for object schemas.")


class ToolParameters(ToolSchema):
    """Top-level parameter container for function tools."""

    type: Literal["object"] = "object"


class ToolFunction(OpenAIBaseModel):
    """OpenAI function specification with enforced best practices."""

    name: str
    description: str
    parameters: ToolParameters = Field(default_factory=ToolParameters)

    @model_validator(mode="before")
    @classmethod
    def _unwrap_openai_spec(cls, data: Any) -> Any:
        if (
            isinstance(data, dict)
            and data.get("type") == "function"
            and isinstance(data.get("function"), dict)
        ):
            return dict(data["function"])
        return data

    @model_validator(mode="before")
    @classmethod
    def _normalize_parameters(cls, data: Any) -> Any:
        if isinstance(data, dict):
            params = data.get("parameters")
            if isinstance(params, dict):
                schema_type = params.get("type")
                has_properties = "properties" in params
                schema_valid = (
                    has_properties
                    and isinstance(schema_type, str)
                    and schema_type in VALID_SCHEMA_TYPES
                )
                if not schema_valid:
                    data = dict(data)
                    data["parameters"] = {"type": "object", "properties": params}
        return data

    @field_validator("parameters", mode="before")
    @classmethod
    def _coerce_parameters(cls, value: Any) -> Any:
        if isinstance(value, dict):
            schema_type = value.get("type")
            has_properties = "properties" in value
            schema_valid = (
                has_properties
                and isinstance(schema_type, str)
                and schema_type in VALID_SCHEMA_TYPES
            )
            if not schema_valid:
                return {"type": "object", "properties": value}
        return value

    @field_validator("name")
    @classmethod
    def _check_name(cls, name: str) -> str:
        if not NAME_PATTERN.match(name):
            raise ValueError(
                "Function names must be 1-64 characters long and snake_case (a-z0-9_)."
            )
        return name

    def model_dump(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        if "exclude_none" not in kwargs:
            kwargs["exclude_none"] = True
        if "by_alias" not in kwargs:
            kwargs["by_alias"] = True
        function_payload = super().model_dump(*args, **kwargs)
        return {"type": "function", "function": function_payload}

    def model_dump_json(self, *args: Any, **kwargs: Any) -> str:
        return json.dumps(self.model_dump(*args, **kwargs), ensure_ascii=False)


ToolSchema.model_rebuild()
