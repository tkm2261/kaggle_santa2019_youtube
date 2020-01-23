import os
import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm

import gurobipy as gp

from logging import getLogger

logger = getLogger(None)


def main():
    logger.info("start")
    n_family = 5000
    n_days = 100
    # n_choice = 10

    days = list(range(n_days, 0, -1))

    df_family = pd.read_csv("../input/family_data.csv", index_col="family_id")
    map_family_size = df_family["n_people"].to_dict()

    with open("mat_cost.pkl", "rb") as f:
        mat_cost = pickle.load(f)
    map_var_x_cost = {(f, d): mat_cost[f, d - 1] for f in range(n_family) for d in days}
    map_var_x_name = {(f, d): f"x_{f}_{d}" for f in range(n_family) for d in days}

    # map_var_x_cost[f, d] family i, day d

    map_cost_ac = {}
    for p in tqdm(range(125, 301)):
        for q in range(125, 301):
            diff = abs(p - q)
            map_cost_ac[p, q] = min(
                max(0, (p - 125.0) / 400.0 * q ** (0.5 + diff / 50.0)), 100000
            )

    map_var_z_cost = {
        (p, q, d): map_cost_ac[p, q]
        for p in range(125, 301)
        for q in range(125, 301)
        for d in days
    }
    map_var_z_name = {
        (p, q, d): f"z_{p}_{q}_{d}"
        for p in range(125, 301)
        for q in range(125, 301)
        for d in days
    }

    model = gp.Model("santa2019")

    logger.info("Define vars")

    var_x = model.addVars(
        map_var_x_cost.keys(),
        obj=map_var_x_cost,
        vtype=gp.GRB.BINARY,
        name=map_var_x_name,
    )

    var_z = model.addVars(
        map_var_z_cost.keys(),
        obj=map_var_z_cost,
        vtype=gp.GRB.BINARY,
        name=map_var_z_name,
    )

    logger.info("Each familiy must attend 1 session")
    model.addConstrs(
        gp.quicksum([var_x[f, d] for d in days]) == 1 for f in range(n_family)
    )

    logger.info("Each day has 125-300 people")
    model.addConstrs(
        gp.quicksum(var_z[p, q, d] for p in range(125, 301) for q in range(125, 301))
        == 1
        for d in tqdm(days)
    )

    logger.info("Knapsack constraints")
    model.addConstrs(
        gp.quicksum(map_family_size[f] * var_x[f, d] for f in range(n_family))
        == gp.quicksum(
            p * var_z[p, q, d] for p in range(125, 301) for q in range(125, 301)
        )
        for d in tqdm(days)
    )

    logger.info("Day-by-day constraints")

    model.addConstrs(
        gp.quicksum(var_z[p, q, d] for p in range(125, 301))
        == gp.quicksum(var_z[q, p, min(d + 1, 100)] for p in range(125, 301))
        for d in tqdm(days)
        for q in range(125, 301)
    )

    logger.info("Output LP file")
    model.write("santa2019_gurobi.lp")

    logger.info("Optimization starts")

    model.optimize()


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

    handler = FileHandler("demo.log", "a")
    handler.setLevel(DEBUG)
    handler.setFormatter(log_fmt)
    logger.setLevel(DEBUG)
    logger.addHandler(handler)

    main()
