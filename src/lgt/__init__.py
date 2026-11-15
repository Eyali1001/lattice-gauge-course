"""lgt: a small library for the course "Labelings, models, and gauge theory".

Names follow GLOSSARY.md. Quick tour::

    from lgt import *
    B = Board.grid(3, 3)                 # a board (glossary 1.1)
    G = U1()                             # a label group (2.1)
    U = random_labeling(G, B, rng)       # a labeling (2.2)
    face_products(G, B, U)               # face products (2.5)
    reframe(G, B, U, h)                  # reframing (3.1)
    tree_normal_form(G, B, U)            # tree normal form (3.8)
    M = Model(B, Z2(), beta=1.0)         # a model (4.5)
    M.exact().average(f)                 # an exact average (4.6)
"""

from .groups import Group, Z2, ZN, U1, S3, SO3, SU2, PositiveReals, standard_face_cost, wilson_cost, TAU
from .board import Board
from .labeling import (
    walk_product, loop_product, face_products, defects, reframe, random_labeling,
    random_reframing, identity_labeling, tree_normal_form, is_consistent,
    all_labelings, all_reframings, class_ids,
)
from .model import Model, Exact
from . import viz

__all__ = [
    "Group", "Z2", "ZN", "U1", "S3", "SO3", "SU2", "PositiveReals", "standard_face_cost", "wilson_cost", "TAU",
    "Board",
    "walk_product", "loop_product", "face_products", "defects", "reframe",
    "random_labeling", "random_reframing", "identity_labeling", "tree_normal_form",
    "is_consistent", "all_labelings", "all_reframings", "class_ids",
    "Model", "Exact", "viz",
]
