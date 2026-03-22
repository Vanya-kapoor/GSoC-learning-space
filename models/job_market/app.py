"""
Solara visualisation for the Job Market Actions model.

Run with:
    solara run app.py
Opens at: http://localhost:8765

No grid — this model has no spatial component.
Visualisation shows employment status and hiring metrics over time.
"""

from mesa.visualization import SolaraViz, make_plot_component

from model import JobMarket

model_params = {
    "n_workers": {
        "type": "SliderInt",
        "value": 20,
        "label": "Number of workers",
        "min": 5,
        "max": 50,
        "step": 5,
    },
    "n_employers": {
        "type": "SliderInt",
        "value": 3,
        "label": "Number of employers",
        "min": 1,
        "max": 10,
        "step": 1,
    },
    "slots_per_employer": {
        "type": "SliderInt",
        "value": 2,
        "label": "Job slots per employer",
        "min": 1,
        "max": 5,
        "step": 1,
    },
    "burnout_threshold": {
        "type": "SliderInt",
        "value": 3,
        "label": "Burnout threshold (failed searches)",
        "min": 1,
        "max": 10,
        "step": 1,
    },
}

# Chart 1: Employment status over time
employment_chart = make_plot_component(
    {"employed": "tab:green", "searching": "tab:blue", "inactive": "tab:red"}
)

# Chart 2: Hiring method comparison
hiring_chart = make_plot_component(
    {
        "jobs_filled_by_employer": "tab:orange",
        "jobs_filled_independently": "tab:purple",
    }
)

model = JobMarket()

page = SolaraViz(
    model,
    components=[employment_chart, hiring_chart],
    model_params=model_params,
    name="Job Market — Mesa Actions demo",
)
page  # noqa: B018
