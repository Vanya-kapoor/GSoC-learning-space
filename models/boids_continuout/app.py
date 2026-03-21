

import mesa.visualization
from model import BoidsFlock


def boid_portrayal(agent):
    return {"color": "#1D9E75", "size": 10}


model_params = {
    "n_boids": {
        "type": "SliderInt",
        "value": 50,
        "label": "Number of boids",
        "min": 10,
        "max": 200,
        "step": 10,
    },
    "vision_radius": {
        "type": "SliderFloat",
        "value": 10.0,
        "label": "Vision radius",
        "min": 2.0,
        "max": 30.0,
        "step": 1.0,
    },
    "min_dist": {
        "type": "SliderFloat",
        "value": 3.0,
        "label": "Separation distance",
        "min": 0.5,
        "max": 10.0,
        "step": 0.5,
    },
    "max_speed": {
        "type": "SliderFloat",
        "value": 2.0,
        "label": "Max speed",
        "min": 0.5,
        "max": 5.0,
        "step": 0.5,
    },
    "w_sep": {
        "type": "SliderFloat",
        "value": 1.5,
        "label": "Separation weight",
        "min": 0.0,
        "max": 5.0,
        "step": 0.5,
    },
    "w_ali": {
        "type": "SliderFloat",
        "value": 1.0,
        "label": "Alignment weight",
        "min": 0.0,
        "max": 5.0,
        "step": 0.5,
    },
    "w_coh": {
        "type": "SliderFloat",
        "value": 1.0,
        "label": "Cohesion weight",
        "min": 0.0,
        "max": 5.0,
        "step": 0.5,
    },
    "seed": 42,
}

page = mesa.visualization.SolaraViz(
    BoidsFlock,
    components=[
        mesa.visualization.make_space_component(boid_portrayal),
        mesa.visualization.make_plot_component(["avg_speed"]),
    ],
    model_params=model_params,
    name="Boids — ContinuousSpace demo",
)
page  # required by Solara
