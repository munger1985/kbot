# Legacy code boundary

This directory contains KBot 3.x implementations that are not part of the
4.0 runtime. They may be retained temporarily as migration references, but
new 4.0 packages must never import from here.

The old Main API, V1 controllers, V1 Agent/Skill chain, old service layer,
SkillRuntime/SkillManager, prompt/database compatibility code, and shared
`dao` domain graph are stored here while their 4.0 replacements are completed.
The already-retired KC and model compatibility shims and old FileProcessor
implementation are also here. After the unified 4.0 acceptance run, this
directory is deleted rather than deployed or maintained as an adapter.
