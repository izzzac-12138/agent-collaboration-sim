"""Resource and ResourcePool for resource management."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Resource:
    """A single resource unit within the simulation.

    Attributes:
        resource_id: Unique identifier (e.g. "R001").
        resource_type: Category of resource (e.g. "energy", "information").
        amount: Quantity of this resource.
        owner: Agent ID that owns this resource, or None if unowned.
        location: Spatial position as (x, y) tuple, or None if not placed.
    """

    resource_id: str
    resource_type: str
    amount: float
    owner: Optional[str] = None
    location: Optional[tuple[float, float]] = None


class ResourcePool:
    """Manages the collection of resources and transfers between agents.

    Maintains an internal registry of all resources, handles allocation,
    transfer between agents, and provides querying facilities.

    Usage::

        pool = ResourcePool()
        res = pool.add_resource("energy", 100.0, owner="agent_1")
        pool.transfer(res.resource_id, "agent_1", "agent_2", 30.0)
    """

    def __init__(self) -> None:
        self._resources: dict[str, Resource] = {}
        self._next_id: int = 0
        self._transfer_log: list[dict] = []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _generate_id(self) -> str:
        """Return the next unique resource ID in the form R001, R002, ..."""
        self._next_id += 1
        return f"R{self._next_id:03d}"

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add_resource(
        self,
        resource_type: str,
        amount: float,
        owner: Optional[str] = None,
        location: Optional[tuple[float, float]] = None,
    ) -> Resource:
        """Create and register a new resource.

        Args:
            resource_type: Category of the resource.
            amount: Quantity (must be >= 0).
            owner: Agent ID that owns the resource, or None.
            location: Optional (x, y) position.

        Returns:
            The newly created :class:`Resource`.

        Raises:
            ValueError: If *amount* is negative.
        """
        if amount < 0:
            raise ValueError("Resource amount must be non-negative.")
        resource_id = self._generate_id()
        resource = Resource(
            resource_id=resource_id,
            resource_type=resource_type,
            amount=amount,
            owner=owner,
            location=location,
        )
        self._resources[resource_id] = resource
        return resource

    def transfer(
        self,
        resource_id: str,
        from_agent: str,
        to_agent: str,
        amount: float,
    ) -> Resource:
        """Transfer a portion of a resource from one agent to another.

        Deducts *amount* from the source resource and creates a new resource
        owned by the receiver.  The original resource's amount is decreased;
        if it reaches zero it is removed from the pool.

        Args:
            resource_id: ID of the resource to transfer from.
            from_agent: Current owner of the resource.
            to_agent: Agent that will receive the resource.
            amount: Quantity to transfer (must be > 0).

        Returns:
            The new :class:`Resource` created for *to_agent*.

        Raises:
            KeyError: If *resource_id* does not exist.
            ValueError: If the source agent does not own the resource,
                or if *amount* is non-positive or exceeds available amount.
        """
        if amount <= 0:
            raise ValueError("Transfer amount must be positive.")

        if resource_id not in self._resources:
            raise KeyError(f"Resource '{resource_id}' not found.")

        source = self._resources[resource_id]

        if source.owner != from_agent:
            raise ValueError(
                f"Agent '{from_agent}' does not own resource '{resource_id}'."
            )

        if amount > source.amount:
            raise ValueError(
                f"Insufficient amount: requested {amount}, available {source.amount}."
            )

        # Deduct from source
        source.amount -= amount
        removed = False
        if source.amount <= 0:
            del self._resources[resource_id]
            removed = True

        # Create resource for receiver
        new_resource = self.add_resource(
            resource_type=source.resource_type,
            amount=amount,
            owner=to_agent,
            location=source.location,
        )

        # Log the transfer
        self._transfer_log.append(
            {
                "resource_id": resource_id,
                "from_agent": from_agent,
                "to_agent": to_agent,
                "amount": amount,
                "new_resource_id": new_resource.resource_id,
                "source_remaining": source.amount if not removed else 0.0,
            }
        )

        return new_resource

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def get_agent_resources(self, agent_id: str) -> list[Resource]:
        """Return all resources currently owned by *agent_id*."""
        return [r for r in self._resources.values() if r.owner == agent_id]

    def get_resource(self, resource_id: str) -> Resource:
        """Look up a single resource by ID.

        Raises:
            KeyError: If *resource_id* is not in the pool.
        """
        if resource_id not in self._resources:
            raise KeyError(f"Resource '{resource_id}' not found.")
        return self._resources[resource_id]

    def get_all_resources(self) -> list[Resource]:
        """Return every resource in the pool."""
        return list(self._resources.values())

    def get_pool_resources(self) -> list[Resource]:
        """Return resources that have no owner (unassigned pool resources)."""
        return [r for r in self._resources.values() if r.owner is None]

    def get_transfer_log(self) -> list[dict]:
        """Return the full transfer history."""
        return list(self._transfer_log)

    def total_by_type(self, agent_id: Optional[str] = None) -> dict[str, float]:
        """Sum resource amounts grouped by type.

        Args:
            agent_id: If provided, only count resources owned by this agent.
                If None, count all resources in the pool.

        Returns:
            Mapping of resource type to total amount.
        """
        totals: dict[str, float] = {}
        for resource in self._resources.values():
            if agent_id is not None and resource.owner != agent_id:
                continue
            totals[resource.resource_type] = (
                totals.get(resource.resource_type, 0.0) + resource.amount
            )
        return totals
