from __future__ import annotations
import warnings
import mesa
import matplotlib
from mesa.experimental.actions import Action, ActionState
import solara
from mesa.visualization import SolaraViz, make_plot_component

class Resource:
   
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
       
        return (self.capacity - self.available) / self.capacity

    @property
    def avg_wait_time(self) -> float:
      
        if not self._queue:
            return 0.0
        return sum(
            self.model.time - a._queued_at
            for a in self._queue
            if hasattr(a, "_queued_at")
        ) / len(self._queue)

    def request(self, action: Action) -> None:
        
        if self.available > 0:
            self._grant(action)
        else:
            action._queued_at = self.model.time
            self._queue.append(action)

    def release(self, action: Action) -> None:
       
        self._active.discard(action)
        self.available += 1
        self._serve_next()

    def remove(self, action: Action) -> None:
      
        try:
            self._queue.remove(action)
        except ValueError:
            pass

    def _grant(self, action):
        if action.state == ActionState.PENDING or action.state == ActionState.INTERRUPTED:
            action.start()
   

    def _serve_next(self) -> None:
        while self._queue and self.available > 0:
            action = self._queue.pop(0)
            if action.agent in action.agent.model.agents:
                self._grant(action)


class InterruptGuard:

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


class HasActions:
    def on_action_start(self, action: Action) -> None:
        pass

    def on_action_complete(self, action: Action) -> None:
        pass

    def on_action_interrupt(self, action: Action, progress: float) -> None:
       
        pass


class SearchForJob(InterruptGuard, Action):
   

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
     
        self.job_pool.request(self)
       
        if isinstance(self.agent, HasActions):
            self.agent.on_action_start(self)

    def on_complete(self):
        
        self.agent.status = "employed"
        self.agent.employed = True
        self.agent.experience += 1.0
        self.job_pool.release(self)
        self.agent.model.jobs_filled += 1
       
        if isinstance(self.agent, HasActions):
            self.agent.on_action_complete(self)

    def on_interrupt(self, progress: float):
      
        self.agent.experience += round(progress, 2)
        self.agent.status = "employed"
        self.agent.employed = True
        self.job_pool.release(self)
        self.agent.model.jobs_filled_by_interruption += 1
      
        if isinstance(self.agent, HasActions):
            self.agent.on_action_interrupt(self, progress)


class BurnOut(InterruptGuard, Action):
  
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


class Worker(HasActions, mesa.Agent):
   

    def __init__(self, model: "JobMarket", skill: float):
        super().__init__(model)
        self.skill = skill
        self.status = "idle"
        self.employed = False
        self.searches = 0
        self.experience = 0.0
        self._failures = 0

   
    def on_action_complete(self, action: Action) -> None:
        
        if isinstance(action, BurnOut):
            return  # inactive — do nothing
        if isinstance(action, SearchForJob) and not self.employed:
          
            self._failures += 1
            if self._failures >= self.model.burnout_threshold:
                self.start_action(BurnOut(self))
            else:
                self.start_action(SearchForJob(self, self.model.job_pool))

    def on_action_interrupt(self, action: Action, progress: float) -> None:
      
        pass

    def begin_search(self):
        self.start_action(SearchForJob(self, self.model.job_pool))  # new instance each time


class Employer(mesa.Agent):
  

    def __init__(self, model: "JobMarket"):
        super().__init__(model)

    def step(self) -> None:
        
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


class IdleDetector:
 

    def __init__(self, model: mesa.Model, threshold: int = 5):
        self.model = model
        self.threshold = threshold
        self._idle_counts: dict[int, int] = {}

    def check(self) -> None:
     
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

class JobMarket(mesa.Model):
    def __init__(
    self,
    n_workers: int = 20,
    n_employers: int = 3,
    total_job_slots: int = None,
    slots_per_employer: int = None,  # 👈 ADD THIS LINE
    burnout_threshold: int = 3,
    detect_idle: bool = False,
    rng=None,
    ):
        super().__init__(rng=rng)
       
        if total_job_slots is None and slots_per_employer is not None:
            total_job_slots = n_employers * slots_per_employer
        elif total_job_slots is None:
            total_job_slots = 6  # default fallback

        self.burnout_threshold = burnout_threshold

     
        self.job_pool = Resource(self, capacity=total_job_slots)

       
        self.jobs_filled = 0
        self.jobs_filled_by_interruption = 0
        self.total_burnouts = 0

     
        for _ in range(n_workers):
            # Slight bias toward mid-skill workers (more realistic distribution)
            skill = round(self.random.uniform(0, 5), 1)
            Worker(self, skill=skill)

      
        for _ in range(n_employers):
            Employer(self)

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

        
        for worker in self.agents_by_type[Worker]:
            worker.begin_search()

    def step(self) -> None:
       
        for employer in self.agents_by_type[Employer]:
            employer.step()

        self.run_for(1.0)

      
        if self._idle_detector:
            self._idle_detector.check()

        self.datacollector.collect(self)


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


EmploymentPlot = make_plot_component(["employed", "searching", "inactive"])

@solara.component
def Page():
    SolaraViz(
        JobMarket(n_workers=20, n_employers=3, slots_per_employer=2, burnout_threshold=3, rng=42),
        model_params={
            "n_workers": solara.SliderInt("Workers", value=20, min=5, max=50),
            "n_employers": solara.SliderInt("Employers", value=3, min=1, max=10),
            "slots_per_employer": solara.SliderInt("Slots per Employer", value=2, min=1, max=5),
            "burnout_threshold": solara.SliderInt("Burnout Threshold", value=3, min=1, max=10),
        },
        components=[EmploymentPlot],
    )