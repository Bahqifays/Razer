"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.18.7
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import preprocess_razer, calculation_razer


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=preprocess_razer,
                inputs="razer_raw",
                outputs="razer_preprocessed",
                name="razer_preprocess_node",
            ),
            node(
                func=calculation_razer,
                inputs=["razer_preprocessed", "parameters"],
                outputs="razer_calculated",
                name="razer_calculated_node",
            ),
        ]
    )