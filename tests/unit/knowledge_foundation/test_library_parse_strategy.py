"""审批档位门槛和库模型角色合同。"""

from uuid import UUID

from pydantic import ValidationError

from platform_core.contracts.knowledge_parsing import LibraryParseModels, parse_strategy_options


def test_parse_strategy_options_follow_available_models():
    for models, expected in (
        ({}, {"TEXT"}),
        ({"vlm": "available"}, {"TEXT"}),
        ({"ocr": "available"}, {"TEXT", "AUTO"}),
        ({"ocr": "available", "vlm": "available"}, {"TEXT", "AUTO", "VISUAL"}),
    ):
        options = parse_strategy_options(models)
        assert [option.strategy for option in options] == ["TEXT", "AUTO", "VISUAL"]
        assert {option.strategy for option in options if option.selectable} == expected
        assert options[-1].unavailable_roles == tuple(role for role in ("ocr", "vlm") if role not in models)


def test_library_model_bindings_do_not_accept_strategy_or_unregistered_roles():
    model = UUID("0193e7d9-09b0-7c74-8479-98ef4650b5da")
    assert LibraryParseModels(ocr=model).model_dump(mode="json") == {"ocr": str(model), "vlm": None}
    for invalid in ({"strategy": "AUTO"}, {"tagging": str(model)}, {"ocr": "not-a-model-id"}):
        try:
            LibraryParseModels.model_validate(invalid)
        except ValidationError:
            continue
        raise AssertionError("建库的解析模型合同接受了不合法字段")


if __name__ == "__main__":
    test_parse_strategy_options_follow_available_models()
    test_library_model_bindings_do_not_accept_strategy_or_unregistered_roles()
