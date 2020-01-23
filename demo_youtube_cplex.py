import os
import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm
from docplex.mp.model import Model

from logging import getLogger

logger = getLogger(None)


def main():
    logger.info("start")
    n_family = 5000
    n_days = 100
    n_choice = 10

    days = list(range(n_days, 0, -1))

    df_family = pd.read_csv("../input/family_data.csv", index_col="family_id")

    map_family_size = df_family["n_people"].to_dict()

    with open("mat_cost.pkl", "rb") as f:
        mat_cost = pickle.load(f)
    map_var_x_cost = {(f, d): mat_cost[f, d - 1] for f in range(n_family) for d in days}

    # map_var_x_cost[f, d] family i, day d

    map_cost_ac = {}
    for i in tqdm(range(125, 301)):
        for j in range(125, 301):
            diff = abs(i - j)
            map_cost_ac[i, j] = min(
                max(0, (i - 125.0) / 400.0 * i ** (0.5 + diff / 50.0)), 100000
            )
    map_var_x_name = {(i, j): f"x_{i}_{j}" for i in range(n_family) for j in days}

    map_var_z_cost = {
        (i, j, d): map_cost_ac[i, j]
        for i in range(125, 301)
        for j in range(125, 301)
        for d in days
    }
    map_var_z_name = {
        (i, j, d): f"z_{i}_{j}_{d}"
        for i in range(125, 301)
        for j in range(125, 301)
        for d in days
    }

    model = Model("santa2019", log_output=True)

    logger.info("Define vars")
    var_x = model.var_dict(
        map_var_x_cost.keys(), model.binary_vartype, name=lambda x: map_var_x_name[x],
    )

    var_z = model.var_dict(
        map_var_z_cost.keys(), model.binary_vartype, name=lambda x: map_var_z_name[x],
    )

    logger.info("Each familiy must attend 1 session")
    model.add_constraints(
        model.sum([var_x[f, d] for d in days]) == 1 for f in range(n_family)
    )

    logger.info("Each day has 125-300 people")

    model.add_constraints(
        model.sum(var_z[p, q, d] for p in range(125, 301) for q in range(125, 301)) == 1
        for d in tqdm(days)
    )

    logger.info("Knapsack constraints")

    model.add_constraints(
        model.sum(map_family_size[f] * var_x[f, d] for f in range(n_family))
        == model.sum(
            p * var_z[p, q, d] for p in range(125, 301) for q in range(125, 301)
        )
        for d in tqdm(days)
    )

    logger.info("Day-by-day constraints")

    model.add_constraints(
        model.sum(var_z[p, q, d] for p in range(125, 301))
        == model.sum(var_z[q, p, min(d + 1, 100)] for p in range(125, 301))
        for d in tqdm(days)
        for q in range(125, 301)
    )
    logger.info("Objective function")

    obj = model.sum(v * var_x[k] for k, v in map_var_x_cost.items()) + model.sum(
        v * var_z[k] for k, v in map_var_z_cost.items()
    )
    model.minimize(obj)

    logger.info("Output LP file")
    model.export_as_lp("santa2019_cplex.lp")

    logger.info("optimization starts")

    model.solve()


if __name__ == "__main__":
    from logging import StreamHandler, DEBUG, Formatter, FileHandler

    log_fmt = Formatter(
        "%(asctime)s %(name)s %(lineno)d [%(levelname)s][%(funcName)s] %(message)s "
    )

    handler = StreamHandler()
    handler.setLevel("INFO")
    handler.setFormatter(log_fmt)
    logger.setLevel("INFO")
    logger.addHandler(handler)

    handler = FileHandler(os.path.basename(os.path.abspath(__file__)) + ".log", "a")
    handler.setLevel(DEBUG)
    handler.setFormatter(log_fmt)
    logger.setLevel(DEBUG)
    logger.addHandler(handler)

    main()
