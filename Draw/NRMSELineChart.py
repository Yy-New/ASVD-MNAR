import os
import pandas as pd
import matplotlib.pyplot as plt

from utils import is_folder_exists


def draw_line(gamma_prop_files_path, gamma_prop_files, threshold_concentration_file, file, save_path):
    fig, ax = plt.subplots(3, 4, figsize=(16, 11), dpi=300)

    labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l']
    MNAR_list = ["50%", "54%", "59%", "63%", "68%", "72%", "77%", "81%", "86%", "90%", "94%", "99%"]
    cnt = 0

    handles = []
    labels_legend = []

    for i in range(3):
        for j in range(4):
            index = i * 4 + j

            result_path = f'{gamma_prop_files_path}/{gamma_prop_files[cnt]}'
            multiple_data = pd.read_excel(result_path, index_col=0)
            my_data = pd.read_excel(result_path.replace("MultipleImputation", "ASVDImputation"), index_col=0)
            data = pd.concat([multiple_data, my_data], axis=0)

            data = data.loc[['rf', 'svd', 'QRILC', 'knn', 'ns_knn', 'iter_svd', 'ASVD-MNAR']]
            x_values = data.columns.values

            markers = ['o', 's', 'D', '^', 'v', 'P', 'h']
            line_styles = ['-', '--', ':', '-.', (0, (3, 1, 1, 1)), (0, (5, 1)), '-']

            for k in range(len(data)):
                line, = ax[i, j].plot(x_values, data.values[k], label=data.index.values[k], marker=markers[k],
                                      linestyle=line_styles[k], linewidth=2)

                if cnt == 0:
                    handles.append(line)
                    labels_legend.append(data.index.values[k])

            ax[i, j].set_title(f"MNAR Proportion: Approx {MNAR_list[cnt]}")
            ax[i, j].text(-0.1, 1.09, labels[index], transform=ax[i, j].transAxes, fontsize=18, ha='left', va='top')
            ax[i, j].set_xlabel('Missing Rate')
            ax[i, j].set_ylabel('NRMSE')

            ax[i, j].grid(True, linestyle='--', alpha=0.6)
            cnt += 1
            if cnt == 1:
                break
        if cnt == 1:
            break

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    fig.suptitle(f"{file}_{threshold_concentration_file}", fontsize=16, y=0.99)

    fig.legend(handles, labels_legend, loc='upper center', ncol=len(labels_legend), fontsize=14, frameon=False, bbox_to_anchor=(0.5, 0.97))

    tmp_path = f'{save_path.replace("MultipleImputation", "ASVD_MNAR")}'
    is_folder_exists(tmp_path)
    plt.savefig(f'{tmp_path}/{file}_{threshold_concentration_file}.svg', dpi=800)

    plt.show()


file_path = "../AverageResults/NRMSE/MultipleImputation"
save_path = file_path.replace("AverageResults", "ResultGraph")
is_folder_exists(save_path)
file_list = os.listdir(file_path)
for file in file_list:
    threshold_concentration_path = f"{file_path}/{file}"
    threshold_concentration_files = os.listdir(threshold_concentration_path)
    for threshold_concentration_file in threshold_concentration_files:
        gamma_prop_files_path = f"{threshold_concentration_path}/{threshold_concentration_file}"
        gamma_prop_files = os.listdir(gamma_prop_files_path)
        draw_line(gamma_prop_files_path, gamma_prop_files, threshold_concentration_file, file, save_path)
