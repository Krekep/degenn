from collections import defaultdict

from experiments.import_experiment_results import import_results
from upload_to_google import work_sheets, table, sheets
import pandas as pd
import os


def _count_statistic(df_list, threshold):
    global count_trained_networks, count_all_time
    mean_nn = 0
    mean_time = 0
    count_success = 0
    for df in df_list:
        count_trained_networks += len(df.index)
        count_all_time += df["train_time"].sum()
        last_loss_value = df["loss"].iloc[-1]
        if last_loss_value <= threshold:
            count_success += 1
            mean_nn += len(df.index)
            mean_time += df["train_time"].sum()
    mean_nn = mean_nn / max(count_success, 1)
    mean_time = mean_time / max(count_success, 1)
    return mean_time, mean_nn, count_success, len(df_list)


if __name__ == "__main__":
    count_trained_networks = 0
    count_all_time = 0
    folder_path = "results"

    data = import_results(folder_path)

    for description, df_list in data.items():
        (
            alg_name,
            threshold,
            distance,
            temperature,
            func_name,
            data_size,
            loss,
            opt,
        ) = description
        mean_time, mean_nn, count_success, all_launches = _count_statistic(
            df_list, float(threshold)
        )
        if all_launches < 20:
            mean_time = -1
            mean_nn = -1
            count_success = -all_launches / 100
        if alg_name == "Random":
            sheets[func_name][(alg_name)][loss][threshold][data_size][
                "Time(s)"
            ].value = round(mean_time, 2)
            sheets[func_name][(alg_name)][loss][threshold][data_size][
                "Count"
            ].value = round(mean_nn, 2)
            sheets[func_name][(alg_name)][loss][threshold][data_size][
                "Launch 20"
            ].value = int(100 * (count_success / all_launches))
        if alg_name == "Ann":
            sheets[func_name][(alg_name, temperature, distance)][loss][threshold][
                data_size
            ]["Time(s)"].value = round(mean_time, 2)
            sheets[func_name][(alg_name, temperature, distance)][loss][threshold][
                data_size
            ]["Count"].value = round(mean_nn, 2)
            sheets[func_name][(alg_name, temperature, distance)][loss][threshold][
                data_size
            ]["Launch 20"].value = int(100 * (count_success / all_launches))
    for key, sheet in sheets.items():
        cell_list = list()
        for _, sh in sheet.items():
            for _, lv in sh.items():
                for _, tv in lv.items():
                    for _, dsv in tv.items():
                        for _, cell in dsv.items():
                            cell_list.append(cell)
        table.worksheet(key.capitalize()).update_cells(cell_list)
    print(
        "Total number of trained neural networks during experiments =",
        count_trained_networks,
    )
    print("Total time(s) during experiments =", count_all_time)
