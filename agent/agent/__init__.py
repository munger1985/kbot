"""Agent package.

Legacy 3.x agents are not imported eagerly.  This keeps the active V2 modules
independently importable while the old Agent/Skill runtime is retired.
"""

__all__: list[str] = []
