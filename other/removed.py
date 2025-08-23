

def plot_bias_variance_bars_varying_n_bags(results_list, name_list, fixed_k, fixed_sampling_rate, show = False, save_name='n_bags_bar_plot'):
    data_by_dataset = defaultdict(list)

    for name, result in zip(name_list, results_list):
        # Only consider mle_bag
        if 'mle_bag' not in name:
            continue

        k_match = re.search(r'_k_(\d+)', name)
        sr_match = re.search(r'sampling_rate_([0-9.]+)', name)
        nb_match = re.search(r'n_bags_(\d+)', name)

        if not (k_match and sr_match and nb_match):
            continue

        k = int(k_match.group(1))
        sr = sr_match.group(1)
        n_bags = int(nb_match.group(1))

        if k != fixed_k or sr != str(fixed_sampling_rate):
            continue

        method_label = f'{n_bags}'

        for dataset, values in result.items():
            mse = values[1]
            bias2 = values[2]
            var = values[3]
            data_by_dataset[dataset].append({
                'method': method_label,
                'mse': mse,
                'bias2': bias2,
                'var': var,
                'n_bags': n_bags
            })

    # Step 2: Sort entries by n_bags
    for dataset in data_by_dataset:
        data_by_dataset[dataset].sort(key=lambda e: e['n_bags'])

    # Step 3: Plot
    dataset_names = sorted(data_by_dataset.keys())
    fig, axes = plt.subplots(len(dataset_names), 1, figsize=(10, 3.5 * len(dataset_names)), sharex=True)
    if len(dataset_names) == 1:
        axes = [axes]

    for ax, dataset in zip(axes, dataset_names):
        entries = data_by_dataset[dataset]
        if not entries:
            continue

        methods = [e['method'] for e in entries]
        mse_vals = [e['mse'] for e in entries]
        bias_vals = [e['bias2'] for e in entries]
        var_vals = [e['var'] for e in entries]

        x = list(range(len(methods)))
        bar_width = 0.6

        ax.bar(x, bias_vals, width=bar_width, color='green', label='Bias²')
        ax.bar(x, var_vals, width=bar_width, bottom=bias_vals, color='red', label='Variance')

        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=45, ha='right')
        ax.set_title(f'MSE Decomposition at k={fixed_k}, rate={fixed_sampling_rate} - {dataset}')
        ax.set_ylabel('MSE')
        ax.grid(axis='y', linestyle='--', alpha=0.4)
        ax.legend()

    plt.tight_layout()
    directory = r'C:\Users\User\PycharmProjects\pythonProject3\LIDstuff\saved_results\plots'
    plt.savefig(directory + '\\' + f'{save_name}.pdf', bbox_inches="tight")
    if show:
        plt.show()