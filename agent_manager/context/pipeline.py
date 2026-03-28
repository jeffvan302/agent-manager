"""Pre-call context pipeline wrapper."""

from __future__ import annotations

from agent_manager.config import RuntimeConfig
from agent_manager.context.assembler import ContextAssembler
from agent_manager.context.sections import PreparedTurn
from agent_manager.types import LoopState


class PreCallPipeline:
    """Thin wrapper around context assembly so later phases can add more steps."""

    def __init__(self, assembler: ContextAssembler | None = None) -> None:
        self.assembler = assembler or ContextAssembler()

    def prepare(self, state: LoopState, config: RuntimeConfig) -> PreparedTurn:
        return self.assembler.prepare(state, config)

