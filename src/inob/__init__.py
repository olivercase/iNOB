"""inob — forward modelling pipeline for vagus nerve OPM-MEG.

Stages: geometry → FEM mesh → sensor array → DUNEuro leadfield.

A single YAML config (configs/default.yaml) drives every stage; the
`inob.cli.pipeline` orchestrator chains them with file-based skip logic.
"""

__version__ = "0.1.0"
__all__ = ["__version__"]
