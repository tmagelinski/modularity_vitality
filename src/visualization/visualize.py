import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def fragFigs(net_name, results, attackType='Initial', abbrv=None, legend=False, save=False):
    plt.figure(0)
    ax1 = plt.subplot(111)
    plt.figure(1)
    ax2 = plt.subplot(111)

    for function_name, (rho, sigma, rho_e) in results.items():
        if abbrv:
            function_name = abbrv[function_name]
        if function_name == 'CD':
            continue

        plt.figure(0)
        plt.plot(rho, sigma, label=function_name)

        plt.figure(1)
        plt.plot(rho_e, sigma, label=function_name)

    plt.figure(0)
    plt.title(attackType + ' Attack on ' + net_name, fontsize=16)
    plt.ylabel(r'$\sigma$', fontsize=16)
    plt.xlabel(r'$\rho$', fontsize=16)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.spines["bottom"].set_visible(False)
    ax1.spines["left"].set_visible(False)
    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)
    grid_lines = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    for i in grid_lines:
        plt.hlines(i, xmin=0, xmax=1, linestyles='--', lw=0.5, color='black', alpha=0.3)
    for i in grid_lines:
        plt.vlines(i, ymin=0, ymax=1, linestyles='--', lw=0.5, color='black', alpha=0.3)

    fig_name = net_name.replace(' ', '_')
    if save:
        plt.savefig(f'../reports/figures/{fig_name}_sig_frag.eps', bbox_inches='tight')

    plt.figure(1)
    if legend:
        plt.legend(fontsize=14)
        plt.legend(loc='center left', bbox_to_anchor=(.65, .62), fontsize=14)
    plt.title(attackType + ' Attack on ' + net_name, fontsize=16)
    plt.ylabel(r'$\sigma$', fontsize=16)
    plt.xlabel(r'$\eta$', fontsize=16)

    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.spines["bottom"].set_visible(False)
    ax2.spines["left"].set_visible(False)
    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)
    for i in grid_lines:
        plt.hlines(i, xmin=0, xmax=1, linestyles='--', lw=0.5, color='black', alpha=0.3)
    for i in grid_lines:
        plt.vlines(i, ymin=0, ymax=1, linestyles='--', lw=0.5, color='black', alpha=0.3)
    if save:
        plt.savefig(f'../reports/figures/{fig_name}_eta_frag.eps', bbox_inches='tight')
    return


def heatmap(net_name, centrality_values, abbrv, save=False):
    result_df = pd.DataFrame(centrality_values)
    result_df.rename(columns=abbrv, inplace=True)
    result_df = result_df.loc[:, ['MV', 'AMV', 'Mas', 'AMC-D', 'CD', 'CHB', 'WMC-D', 'Deg']]
    result_df['MV'] = result_df['MV'].apply(lambda x: -1 * x)  # account for negative-first attack
    metric_corr = result_df.corr(method='kendall')
    cmap = sns.diverging_palette(10, 240, sep=20, as_cmap=True)
    plt.figure(0)
    sns.heatmap(metric_corr, cmap=cmap, annot=True, cbar_kws={'label': 'Kendall-Tau Correlation'}, vmin=-1, vmax=1)
    plt.title(net_name + ' Correlation')
    fig_name = net_name.replace(' ', '_')
    if save:
        plt.savefig(f'../reports/figures/{fig_name}_heatmap_w_cd.eps', bbox_inches='tight')

    metric_corr.drop('CD', axis=0, inplace=True)
    metric_corr.drop('CD', axis=1, inplace=True)
    plt.figure(1)
    sns.heatmap(metric_corr, cmap=cmap, annot=True, cbar_kws={'label': 'Kendall-Tau Correlation'}, vmin=-1, vmax=1)
    plt.title(net_name + ' Correlation')
    fig_name = net_name.replace(' ', '_')
    if save:
        plt.savefig(f'../reports/figures/{fig_name}_heatmap_no_cd.eps', bbox_inches='tight')
    return
