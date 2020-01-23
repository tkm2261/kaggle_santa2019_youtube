import os
import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm

from pulp import LpVariable, LpBinary, LpContinuous, LpProblem
from pulp import LpMinimize, lpSum, COIN_CMD

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

    model = LpProblem("santa2019", LpMinimize)

    logger.info("Define vars")
    var_x = {k: LpVariable(f"x_{k}", cat=LpBinary) for k in map_var_x_cost}
    var_z = {k: LpVariable(f"z_{k}", cat=LpBinary) for k in map_var_z_cost}

    logger.info("Each familiy must attend 1 session")
    for f in range(n_family):
        model += lpSum([var_x[f, d] for d in days]) == 1

    logger.info("Each day has 125-300 people")
    for d in tqdm(days):
        model += (
            lpSum(var_z[p, q, d] for p in range(125, 301) for q in range(125, 301)) == 1
        )

    logger.info("Knapsack constraints")
    for d in tqdm(days):
        model += lpSum(
            map_family_size[f] * var_x[f, d] for f in range(n_family)
        ) == lpSum(p * var_z[p, q, d] for p in range(125, 301) for q in range(125, 301))

    logger.info("Day-by-day constraints")
    for d in tqdm(days):
        for q in range(125, 301):
            model += lpSum(var_z[p, q, d] for p in range(125, 301)) == lpSum(
                var_z[q, p, min(d + 1, 100)] for p in range(125, 301)
            )

    logger.info("Objective function")
    model += lpSum(v * var_x[k] for k, v in map_var_x_cost.items()) + lpSum(
        v * var_z[k] for k, v in map_var_z_cost.items()
    )

    logger.info("Output LP file")
    model.writeLP("santa2019_pulp.lp")

    logger.info("optimization starts")

    solver = COIN_CMD(presolve=1, threads=8, maxSeconds=3600, msg=1)

    model.solve(solver)


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
