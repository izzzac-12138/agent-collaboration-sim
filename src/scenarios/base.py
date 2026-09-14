"""Abstract base class for scenario plugins."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass


class ScenarioBase(ABC):
    """Base class that all scenario plugins must inherit from.

    A scenario defines the environment, objectives, and completion criteria
    for a multi-agent simulation run.
    """

    @abstractmethod
    def setup(self, model: Any) -> None:
        """Initialise the scenario state before agents begin acting.

        Args:
            model: The shared simulation model / state object that agents
                will interact with throughout the scenario.
        """

    @abstractmethod
    def agent_decide(self, agent: Any, model: Any) -> list[dict]:
        """Produce one agent's set of decisions for the current tick.

        Args:
            agent: The agent requesting a decision.
            model: The current simulation model / state.

        Returns:
            A list of decision dictionaries, each describing an action
            the agent wishes to take.
        """

    @abstractmethod
    def check_completion(self, model: Any) -> bool:
        """Evaluate whether the scenario has reached its end condition.

        Args:
            model: The current simulation model / state.

        Returns:
            ``True`` if the scenario is complete, ``False`` otherwise.
        """

    def get_name(self) -> str:
        """Return a short human-readable identifier for this scenario.

        The default implementation returns the class name; subclasses may
        override this to provide a more descriptive label.
        """
        return self.__class__.__name__

    def get_description(self) -> str:
        """Return a longer human-readable description of the scenario.

        Subclasses should override this to document the scenario's
        narrative, goals, and any notable constraints.
        """
        return ""
