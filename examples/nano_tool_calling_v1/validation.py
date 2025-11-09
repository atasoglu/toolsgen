from typing import Any


def _validate_schema_recursively(schema: dict[str, Any]) -> bool:
    """Recursively validate JSON Schema properties.

    OpenAI requires that array types must have 'items' field defined.
    """
    if not isinstance(schema, dict):
        return True

    # Check if this is an array type
    schema_type = schema.get("type")
    if schema_type == "array":
        # Array must have items field
        if "items" not in schema:
            return False
        # Recursively validate items
        if not _validate_schema_recursively(schema.get("items", {})):
            return False
    elif isinstance(schema_type, list) and "array" in schema_type:
        # Handle union types like ["array", "null"]
        if "items" not in schema:
            return False
        if not _validate_schema_recursively(schema.get("items", {})):
            return False

    # Check properties recursively
    if "properties" in schema:
        properties = schema.get("properties", {})
        if isinstance(properties, dict):
            for prop_schema in properties.values():
                if not _validate_schema_recursively(prop_schema):
                    return False

    # Check items recursively (for nested arrays)
    if "items" in schema:
        items = schema.get("items")
        if isinstance(items, dict):
            if not _validate_schema_recursively(items):
                return False

    # Check additionalProperties if it's a schema
    if "additionalProperties" in schema:
        add_props = schema.get("additionalProperties")
        if isinstance(add_props, dict):
            if not _validate_schema_recursively(add_props):
                return False

    return True


def validate_json_schema(tool: dict[str, Any]) -> bool:
    """Validate OpenAI tool schema format.

    Expected format:

    ```json
    {
        "type": "function",
        "function": {
            "name": "function_name",
            "description": "function description",
            "parameters": {"type": "object", "properties": {...}}
        }
    }
    ```
    Also validates that all array types have 'items' field defined.
    """
    try:
        # Check top-level structure
        if not isinstance(tool, dict):
            return False

        if tool.get("type") != "function":
            return False

        function = tool.get("function")
        if not isinstance(function, dict):
            return False

        # Check required function fields
        if "name" not in function or not isinstance(function["name"], str):
            return False

        if "description" not in function or not isinstance(
            function["description"], str
        ):
            return False

        # Parameters are optional, but if present must be a dict
        if "parameters" in function:
            params = function["parameters"]
            if not isinstance(params, dict):
                return False

            # If parameters exist, should have type: object
            if params.get("type") != "object":
                return False

            # Recursively validate the schema for array types
            if not _validate_schema_recursively(params):
                return False

        return True

    except Exception:
        return False
