"""
Job Market Model — Mesa 4.0 Actions demo

Demonstrates mesa.experimental.actions with real patterns:

    Action subclassing       — on_start, on_complete, on_interrupt
    Callable duration        — skill-dependent search time
    Interruption             — employer hires mid-search worker
    Non-interruptible action — BurnOut cannot be cancelled once started
    Progress tracking        — partial experience from interrupted search
    Resource class           — job slots with capacity + waiting queue
    Interrupt guard          — blocks nested interrupts
    Idle detection           — warns if agent sits without action
    Agent-level hooks        — on_action_complete centralises decision logic

Scenario
--------
Workers search for jobs. Search duration varies by skill — skilled workers
find jobs faster. Employers have limited job slots managed by a Resource.
When a slot opens, the employer interrupts the highest-skill searching worker
and hires them immediately. Workers who fail too many times burn out.

Run headless:
    python model.py

Run with Solara UI:
    solara run app.py
"""

from __future__ import annotations

import warnings

import mesa
from mesa.experimental.actions import Action, ActionState


# ------------------------------------------------------------------ #
# Resource — shared object with capacity + waiting queue
# (proposed addition to mesa.experimental.actions — EwoutH #3304)
# ------------------------------------------------------------------ #

class Resource:
    """A shared object with limited capacity and a waiting queue.

    Covers the recurring pattern: factory machines, hospital beds,
    phone lines, job slots. Multiple agents compete for limited access.

    Actions are queued (not agents) — the Action already carries the
    agent reference, duration, and hooks. When capacity frees up,
    Resource calls action.start() automatically.

    Usage:
        slot = Resource(model, capacity=2)
        slot.request(FindJob(worker))   # starts or queues
        # on action complete: call slot.release(action)
    """

    def __init__(self, model, capacity: int = 1):
        self.model = model
        self.capacity = capacity
        self.available = capacity
        self._queue: list[Action] = []
        self._active: set[Action] = set()

    @property
    def queue_length(self) -> int:
        return len(self._queue)

    @property
    def utilization(self) -> float:
        """Fraction of capacity currently in use."""
        return (self.capacity - self.available) / self.capacity

    @property
    def avg_wait_time(self) -> float:
        """Average wait time across all queued actions."""
        if not self._queue:
            return 0.0
        return sum(
            self.model.time - a._queued_at
            for a in self._queue
            if hasattr(a, "_queued_at")
        ) / len(self._queue)

    def request(self, action: Action) -> None:
        """Request access. Grants immediately or queues."""
        if self.available > 0:
            self._grant(action)
        else:
            action._queued_at = self.model.time
            self._queue.append(action)

    def release(self, action: Action) -> None:
        """Release a slot. Serve next in queue automatically."""
        self._active.discard(action)
        self.available += 1
        self._serve_next()

    def remove(self, action: Action) -> None:
        """Remove action from queue (agent gave up waiting)."""
        try:
            self._queue.remove(action)
        except ValueError:
            pass

    def _grant(self, action: Action) -> None:
        self.available -= 1
        self._active.add(action)
        action.start()

    def _serve_next(self) -> None:
        while self._queue and self.available > 0:
            action = self._queue.pop(0)
            if action.agent in action.agent.model.agents:
                self._grant(action)


# ------------------------------------------------------------------ #
# Interrupt guard mixin
# (your suggestion in mesa/mesa #3304 — blocks nested interrupts)
# ------------------------------------------------------------------ #

class InterruptGuard:
    """Mixin that prevents nested interrupts on an Action.

    If interrupt() is called while on_interrupt() is still executing,
    the nested call is silently ignored. Prevents infinite recursion
    in patterns like: interrupt → on_interrupt → start_action → interrupt.

    Usage:
        class MyAction(InterruptGuard, Action):
            ...
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._is_interrupting: bool = False

    def interrupt(self) -> bool:
        if self._is_interrupting:
            return False
        self._is_interrupting = True
        try:
            return super().interrupt()
        finally:
            self._is_interrupting = False


# ------------------------------------------------------------------ #
# HasActions mixin
# (your suggestion in mesa/mesa PR #3591 — agent-level hooks)
# ------------------------------------------------------------------ #

class HasActions:
    """Mixin for agents that use the Actions system.

    Adds three hooks called automatically after the corresponding
    Action hooks. Decision logic lives once on the agent, not
    scattered across every action's on_complete.

    Usage:
        class Worker(HasActions, mesa.Agent):
            def on_action_complete(self, action):
                self.decide_next()

    Separates naming clearly from Action.on_x hooks:
        Action.on_complete      — what the action does
        HasActions.on_action_complete — what the agent does next
    """

    def on_action_start(self, action: Action) -> None:
        """Called after action starts. Override for logging/setup."""
        pass

    def on_action_complete(self, action: Action) -> None:
        """Called after action completes normally.
        Override to implement the agent's behavioral loop."""
        pass

    def on_action_interrupt(self, action: Action, progress: float) -> None:
        """Called after action is interrupted.
        Override for interrupt-specific agent decisions."""
        pass


# ------------------------------------------------------------------ #
# Actions
# ------------------------------------------------------------------ #

class SearchForJob(InterruptGuard, Action):
    """Worker searches for a job.

    Duration depends on skill — higher skill = faster search.
    Interruptible — employer can hire worker mid-search.
    Partial experience gained even if interrupted.
    """

    def __init__(self, worker: "Worker", job_pool: Resource):
        super().__init__(
            worker,
            duration=lambda a: max(1.0, 10.0 - a.skill * 2.0),
            priority=lambda a: a.skill,
            interruptible=True,
            name="SearchForJob",
        )
        self.job_pool = job_pool

    def on_start(self):
        self.agent.status = "searching"
        self.agent.searches += 1
        # Request a job slot — queues if none available
        self.job_pool.request(self)
        # Fire agent hook (HasActions)
        if isinstance(self.agent, HasActions):
            self.agent.on_action_start(self)

    def on_complete(self):
        """Worker got a job slot and completed the process."""
        self.agent.status = "employed"
        self.agent.employed = True
        self.agent.experience += 1.0
        self.job_pool.release(self)
        self.agent.model.jobs_filled += 1
        # Fire agent hook
        if isinstance(self.agent, HasActions):
            self.agent.on_action_complete(self)

    def on_interrupt(self, progress: float):
        """Employer hired this worker mid-search.

        progress = fraction of search completed (0.0–1.0)
        Experience proportional to progress gained.
        """
        self.agent.experience += round(progress, 2)
        self.agent.status = "employed"
        self.agent.employed = True
        self.job_pool.release(self)
        self.agent.model.jobs_filled_by_interruption += 1
        # Fire agent hook
        if isinstance(self.agent, HasActions):
            self.agent.on_action_interrupt(self, progress)


class BurnOut(InterruptGuard, Action):
    """Worker gives up after too many failed searches.

    Non-interruptible — once burning out, cannot be hired.
    """

    def __init__(self, worker: "Worker"):
        super().__init__(
            worker,
            duration=1.0,
            priority=0.0,
            interruptible=False,
            name="BurnOut",
        )

    def on_complete(self):
        self.agent.status = "inactive"
        self.agent.model.total_burnouts += 1
        if isinstance(self.agent, HasActions):
            self.agent.on_action_complete(self)


# ------------------------------------------------------------------ #
# Agents
# ------------------------------------------------------------------ #

class Worker(HasActions, mesa.Agent):
    """
    Worker agent that searches for jobs using the Actions system.

    Inherits HasActions to centralise decision logic in
    on_action_complete — no decision code in individual actions.

    Attributes:
        skill (float)      : 0–5, affects search speed and hire priority
        status (str)       : idle | searching | employed | inactive
        employed (bool)    : True if currently employed
        searches (int)     : total job searches started
        experience (float) : accumulated (including partial from interrupts)
        _failures (int)    : consecutive failed searches
    """

    def __init__(self, model: "JobMarket", skill: float):
        super().__init__(model)
        self.skill = skill
        self.status = "idle"
        self.employed = False
        self.searches = 0
        self.experience = 0.0
        self._failures = 0

    # HasActions hook — all decision logic in one place
    def on_action_complete(self, action: Action) -> None:
        """Called after any action completes. Decide what to do next."""
        if isinstance(action, BurnOut):
            return  # inactive — do nothing
        if isinstance(action, SearchForJob) and not self.employed:
            # Search completed but not hired — try again or burn out
            self._failures += 1
            if self._failures >= self.model.burnout_threshold:
                self.start_action(BurnOut(self))
            else:
                self.start_action(SearchForJob(self, self.model.job_pool))

    def on_action_interrupt(self, action: Action, progress: float) -> None:
        """Called when action is interrupted — employer hired us."""
        # Already handled in SearchForJob.on_interrupt
        pass

    def begin_search(self) -> None:
        """Start first search."""
        if self.current_action is None and not self.employed:
            self.start_action(SearchForJob(self, self.model.job_pool))


class Employer(mesa.Agent):
    """
    Employer that actively interrupts searching workers to fill slots.

    Uses the Resource's queue naturally — workers who request a slot
    but find it full are served when the slot frees up. The employer
    also actively interrupts high-skill workers mid-search.
    """

    def __init__(self, model: "JobMarket"):
        super().__init__(model)

    def step(self) -> None:
        """Interrupt highest-skill searching worker if slots available."""
        if self.model.job_pool.available == 0:
            return

        searching = [
            w for w in self.model.agents_by_type[Worker]
            if w.current_action is not None
            and isinstance(w.current_action, SearchForJob)
            and w.current_action.state == ActionState.ACTIVE
            and not w.employed
        ]

        if not searching:
            return

        best = max(searching, key=lambda w: w.skill)
        best.current_action.interrupt()


# ------------------------------------------------------------------ #
# Idle detection utility
# (your suggestion — EwoutH asked "maybe detect idle agents?")
# ------------------------------------------------------------------ #

class IdleDetector:
    """Opt-in idle agent detection for models using Actions.

    Warns when an agent's current_action is None for more than
    `threshold` consecutive steps. Zero overhead when disabled.

    Usage:
        detector = IdleDetector(model, threshold=3)
        # call detector.check() each model step
    """

    def __init__(self, model: mesa.Model, threshold: int = 5):
        self.model = model
        self.threshold = threshold
        self._idle_counts: dict[int, int] = {}

    def check(self) -> None:
        """Check all agents for idleness. Call once per step."""
        for agent in self.model.agents:
            if not hasattr(agent, "current_action"):
                continue
            uid = agent.unique_id
            if agent.current_action is None:
                count = self._idle_counts.get(uid, 0) + 1
                self._idle_counts[uid] = count
                if count >= self.threshold:
                    warnings.warn(
                        f"Agent {uid} ({type(agent).__name__}) idle "
                        f"for {count} consecutive steps.",
                        stacklevel=2,
                    )
            else:
                self._idle_counts[uid] = 0


# ------------------------------------------------------------------ #
# Model
# ------------------------------------------------------------------ #

class JobMarket(mesa.Model):
    """
    Job Market simulation using Mesa 4.0 Actions.

    Patterns demonstrated:
        - Action subclassing with on_start/on_complete/on_interrupt
        - Callable duration (skill-dependent search time)
        - Interruption (employer hires mid-search)
        - Non-interruptible action (BurnOut)
        - Resource class (job slots with queue)
        - InterruptGuard mixin (blocks nested interrupts)
        - HasActions mixin (agent-level behavioral hooks)
        - IdleDetector (opt-in idle agent warnings)

    Parameters
    ----------
    n_workers          : int   — number of worker agents
    n_employers        : int   — number of employer agents
    total_job_slots    : int   — total job slots across all employers
    burnout_threshold  : int   — failed searches before burnout
    detect_idle        : bool  — enable idle agent warnings
    rng                : int   — random seed
    """

    def __init__(
        self,
        n_workers: int = 20,
        n_employers: int = 3,
        total_job_slots: int = 6,
        burnout_threshold: int = 3,
        detect_idle: bool = False,
        rng=None,
    ):
        super().__init__(rng=rng)

        self.burnout_threshold = burnout_threshold

        # Shared job pool — Resource with capacity
        self.job_pool = Resource(self, capacity=total_job_slots)

        # Metrics
        self.jobs_filled = 0
        self.jobs_filled_by_interruption = 0
        self.total_burnouts = 0

        # Create workers with varying skill (0–5)
        for _ in range(n_workers):
            skill = round(self.random.uniform(0, 5), 1)
            Worker(self, skill=skill)

        # Create employers
        for _ in range(n_employers):
            Employer(self)

        # Optional idle detection
        self._idle_detector = IdleDetector(self, threshold=3) if detect_idle else None

        self.datacollector = mesa.DataCollector(
            model_reporters={
                "employed": lambda m: sum(
                    1 for w in m.agents_by_type[Worker] if w.employed
                ),
                "searching": lambda m: sum(
                    1 for w in m.agents_by_type[Worker]
                    if w.status == "searching"
                ),
                "inactive": lambda m: sum(
                    1 for w in m.agents_by_type[Worker]
                    if w.status == "inactive"
                ),
                "jobs_filled_by_interruption": "jobs_filled_by_interruption",
                "jobs_filled": "jobs_filled",
                "total_burnouts": "total_burnouts",
                "queue_length": lambda m: m.job_pool.queue_length,
                "slot_utilization": lambda m: round(m.job_pool.utilization, 2),
            }
        )
        self.datacollector.collect(self)

        # All workers start searching immediately
        for worker in self.agents_by_type[Worker]:
            worker.begin_search()

    def step(self) -> None:
        # Employers act first — interrupt high-skill searchers
        for employer in self.agents_by_type[Employer]:
            employer.step()

        # Advance event scheduler by 1 tick
        # This fires any scheduled action completions
        self.run_for(1.0)

        # Optional idle detection
        if self._idle_detector:
            self._idle_detector.check()

        self.datacollector.collect(self)


# ------------------------------------------------------------------ #
# Headless run
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    model = JobMarket(n_workers=20, n_employers=3, total_job_slots=6, rng=42)

    print(
        f"{'Step':<6} {'Employed':<10} {'Searching':<12} "
        f"{'Inactive':<10} {'By Interrupt':<14} {'Queue'}"
    )
    print("-" * 65)

    for step in range(15):
        model.step()
        df = model.datacollector.get_model_vars_dataframe()
        row = df.iloc[-1]
        print(
            f"{step+1:<6} {int(row['employed']):<10} {int(row['searching']):<12} "
            f"{int(row['inactive']):<10} {int(row['jobs_filled_by_interruption']):<14} "
            f"{int(row['queue_length'])}"
        )

    print(f"\nTotal burnouts:            {model.total_burnouts}")
    print(f"Jobs filled by interrupt:  {model.jobs_filled_by_interruption}")
    print(f"Jobs filled via queue:     {model.jobs_filled - model.jobs_filled_by_interruption}")
    print(f"Slot utilization:          {model.job_pool.utilization:.0%}")
