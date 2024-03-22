from pyparsing import col

from matplotlib_setting import set_figure_art
import numpy as np
import pandas as pd
from typing import Union
# import scienceplots
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.ticker as mticker
set_figure_art()
plt.style.context(['ieee'])
import matplotlib.dates as mdates
from matplotlib.ticker import FormatStrFormatter
# pd.options.plotting.backend = "matplotlib"
matplotlib.use('TkAgg')


def line_plot(model_variables, model_variable2=None, model_variable3=None, y_label=None):
    fig, axs = plt.subplots(1
                            , figsize=(4, 2.5)
                            )
    model_variables.plot(ax=axs, color='b')
    if model_variable2 is not None:
        model_variable2.plot(ax=axs, color='g', label=model_variables.__neme__)

    if model_variable3 is not None:
        model_variable3.plot(ax=axs, color='r')

    axs.set_xlabel('Time')
    axs.set_ylabel(f'{y_label} [kWh]')
    fig.show()


def step_plot(df: Union[pd.DataFrame, pd.Series] = None, df2: pd.Series = None, title: str = '', left_y_label: str = '', right_y_label: str = ''):
    fig, ax = plt.subplots(1, figsize=[5, 1.5])
    colors = {
        'HESS': 'red',
        'EV': 'green',
        'BESS': 'blue',
        'Solar irradiance': 'green',
        'House temperature': 'red',
        'Ambient temperature': 'blue',
        'Electricity': 'blue',
        'Heat': 'red',
        'Gas': 'green',
        'P_ev_ch': 'blue',
        'P_ev_dch': 'red',
        'ev_soe': 'green',
        'P_ess_ch': 'blue',
        'P_ess_dch': 'red',
        'ess_soe': 'green',
        'h_tss_ch': 'blue',
        'h_tss_dch': 'red',
        'tss_soe': 'green',
        'p_buy': 'blue',
        'p_sell': 'red',
        'p_import': 'blue',
        'p_export': 'red',
        'h_buy': 'blue',
        'h_sell': 'red',
        'h_import': 'blue',
        'h_export': 'red'

    }

    if isinstance(df, pd.Series):
        ax.step(df.index, df.values, '--', where='mid', linewidth=1, label=df.name, color=colors[df.name])
    else:
        for column in df.columns:
            ax.step(df.index, df[column], '--', where='mid', linewidth=1, label=column, color=colors[column])

    if df2 is not None:
        ax2 = ax.twinx()
        # for column in df2.columns:
        ax2.step(df2.index, df2.values, '--', where='mid', linewidth=1, label=df2.name, color=colors[df2.name])
        ax2.legend(ncol=5,
                   bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower right")
        ax2.set_ylabel(right_y_label, color=colors[df2.name])
        ax2.tick_params(axis='y', labelcolor=colors[df2.name])
        ax2.set_ylim([df2.min() * 0.95, df2.max() * 1.05])

    plt.xticks(df.index, pd.to_datetime(df.index).strftime('%H'))
    ax.xaxis.set_major_locator(mticker.MultipleLocator(8))
    ax.tick_params(which='major', rotation=0)
    ax.set_xlabel('Time')

    # Set the y-axis label and legend
    ax.set_ylabel(left_y_label)
    ax.set_ylim([df.min().min() * 1.05, df.max().max() * 1.05])
    ax.legend(ncol=5,
              bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left")
    ax.grid(True)
    # ax.title.set_text(title)
    fig.set_tight_layout(True)
    fig.show()

    return fig



def stacked_bar(df: pd.DataFrame, df2: pd.DataFrame = None,
                title: str = "", y_label: str = "", fig_ax: tuple = None,
                ):
    tech_color = {
        'HP': '#15B01A',
        'ST': '#069AF3',
        'HESS': '#00FFFF',
        'PV': '#FFFF14',
        'EV': '#13EAC9',
        'mCHP': '#E6DAA6',
        'TSEL': '#A9561E',
        'BESS': '#7E1E9C',
        'SHD': '#E50000',
        'DHW': '#030764',
        'NFEL': '#E6E6F6',
        'NGD': '#C875C4',
        'p_import': '#0000FF',
        'p_export': '#0000FF',
        'h_import': '#FA8072',
        'h_export': '#FA8072',
        'g_import': '#C0C0C0'
    }

    if fig_ax is None:
        fig, ax = plt.subplots(figsize=[5, 1.5])

    else:
        fig, ax = fig_ax
    # Plot the first dataframe
    handles1, labels1 = [], []
    for i, column in enumerate(df.columns):
        handle = ax.bar(df.index, df[column], bottom=df.iloc[:, : i].sum(axis=1),
                        color=tech_color[column]
                        )
        handles1.append(handle)
        labels1.append(column)
    # ax.plot(df.index, df.sum(axis=1), '--', linewidth=0.5, label='Total', color='grey', alpha=1)


    # Plot the second dataframe
    handles2, labels2 = [], []
    if df2 is not None:
        for i, column in enumerate(df2.columns):
            handle = ax.bar(df2.index, df2[column], bottom=df2.iloc[:, : i].sum(axis=1),
                            color=tech_color[column]
                            )
            handles2.append(handle)
            labels2.append(column)
        # ax.plot(df2.index, df2.sum(axis=1), '--', linewidth=0.5, color='grey', alpha=1)

    handles = handles1 + handles2 # + [ax.lines[0]]
    labels = labels1 + labels2 # + ['Total']

    if fig_ax is None:
        # Combine the legend handles and labels

        # Remove duplicate labels
        unique_labels = []
        unique_handles = []
        for i, label in enumerate(labels):
            if label not in unique_labels:
                unique_labels.append(label)
                unique_handles.append(handles[i])

        # Set the x-axis ticks and labels
        plt.xticks(df.index, pd.to_datetime(df.index).strftime('%H'))
        ax.xaxis.set_major_locator(mticker.MultipleLocator(8))
        ax.tick_params(which='major', rotation=0)
        ax.set_xlabel('Time')

        # Set the y-axis label and legend
        ax.set_ylabel(y_label)
        ax.legend(handles=unique_handles,
                  bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left",
                  mode="expand", borderaxespad=0,
                  fancybox=True, labels=unique_labels, ncol=4)
        if df2 is not None:
            ax.set_ylim([df2.sum(axis=1).min() * 1.05, df.sum(axis=1).max() * 1.05])
        else:
            ax.set_ylim([0, df.sum(axis=1).max() * 1.05])
        # ax.title.set_text(title)
        # ax.title.set_fontsize(12)
        # fig.suptitle(title, fontweight='bold')
        # Show the plot
        ax.yaxis.set_label_coords(-0.1, 0.5)
        ax.grid(True)
        fig.set_tight_layout(True)
        if fig_ax is None:
            fig.show()
        return fig
        # fig.savefig(f'hems_results/{title}.pdf')
    else:
        return handles, labels

def aggregate_stacked_bar(df: pd.DataFrame, df2: pd.DataFrame = None,
                          df3: pd.DataFrame = None, title1: str = 'heat [kW]',
                          title2: str = "heat [kW]", y_label: str = ""):
    if df3 is not None:
        fig, ax = plt.subplots(2, figsize=[5, 3], sharex=True, gridspec_kw={'height_ratios': [1, 0.5]})
        handles1, labels1 = stacked_bar(df, df2=df2, fig_ax=(fig, ax[0]))
        handles2, labels2 = stacked_bar(df3, fig_ax=(fig, ax[1]))

        handles = handles1 + handles2
        labels = labels1 + labels2

        # Remove duplicate labels
        unique_labels = []
        unique_handles = []
        for i, label in enumerate(labels):
            if label not in unique_labels:
                unique_labels.append(label)
                unique_handles.append(handles[i])
        plt.xticks(df.index, pd.to_datetime(df.index).strftime('%H'))
        ax[1].xaxis.set_major_locator(mticker.MultipleLocator(8))
        ax[1].tick_params(which='major', rotation=0)
        ax[1].set_xlabel('Time')

        # Set the y-axis label and legend
        ax[0].set_ylabel(y_label)
        # ax[1].set_ylabel(y_label)
        ax[0].yaxis.set_label_coords(-0.07, 0)
        fig.align_ylabels()
        ax[1].legend(handles=unique_handles,
                     bbox_to_anchor=(0, -0.9, 1, 0.2), loc="upper left",
                     mode="expand", borderaxespad=0,
                     fancybox=True, labels=unique_labels, ncol=4)
        ax[0].set_ylim([df2.sum(axis=1).min() * 1.05, df.sum(axis=1).max() * 1.05])
        ax[1].set_ylim([0, df.sum(axis=1).max() * 1.05])
        ax[0].title.set_text(title1)
        ax[1].title.set_text(title2)
        ax[0].grid(True)
        ax[1].grid(True)
        fig.set_tight_layout(True)
        fig.show()
        return fig

    else:
        fig = stacked_bar(df, df2=df2)
        return fig
