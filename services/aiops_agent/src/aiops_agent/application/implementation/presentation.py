"""按结构区分说明与命令；人工审批的 SQL 草案仍是可复制文本，不授予执行权。"""


def is_instruction(command):
    return str(command.get("executor") or command.get("command_type") or "").upper() == "MANUAL"


def step_groups(step):
    for key, command_label, note_label, note_key in (
        ("commands", "实施命令", "人工确认项", None),
        ("verification_commands", "验证命令", "验证要点", "verification_notes"),
        ("rollback", "回退命令", "停止或回退条件", "rollback_conditions"),
    ):
        commands = step.get(key) or ()
        notes = [item for item in commands if is_instruction(item)]
        notes += [{"title": note_label, "content": value} for value in step.get(note_key, ())] if note_key else []
        yield note_label, notes, True
        yield command_label, [item for item in commands if not is_instruction(item)], False
