import os
import wandb
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd
import json
import numpy as np

def get_run_ids_by_tag(tag):
    runs = api.runs(f"{wandb_entity}/{wandb_project}")
    tagged_run_ids = [run.id for run in runs if tag in run.tags]
    return tagged_run_ids

def download_artifacts(run_id, folder):
    run = api.run(f"{wandb_entity}/{wandb_project}/{run_id}")
    artifacts = run.logged_artifacts()
    for artifact in artifacts:
        artifact_dir = artifact.download(root=folder)
        print(f"Downloaded {artifact.name} to {artifact_dir}")

def fetch_table(run, table_key, folder):
    table_path = os.path.join(folder, f"{table_key}.table.json")
    if os.path.exists(table_path):
        table = json.load(open(table_path))
        df = pd.DataFrame(table['data'], columns= table['columns'])
        return df
        
    else:
        print(f"Table {table_key} not found in folder {folder}.")
        return None

def plot_tsne(run_id, table_key, folder, plot_size=(8, 8)):
    run = api.run(f"{wandb_entity}/{wandb_project}/{run_id}")
    df = fetch_table(run, table_key, folder)

    if df is not None:
        embeddings = df['Embedding'].tolist()
        embeddings = np.array(embeddings)
        writers = df['Writer'].tolist()

        tsne = TSNE(n_components=2, perplexity=5, random_state=42)
        tsne_results = tsne.fit_transform(embeddings)


        #fig, axs = plt.subplots(figsize=(8, 8), nrows=1)
        
       # axs[0].scatter(tsne_results[:, 0], tsne_results[:, 1], c=writers)
       # axs[0].set_title("LLE Embedding of Swiss Roll")
        

        plt.figure(figsize=plot_size)
        scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=writers, s=35, alpha=0.8) #, cmap='viridis', s=5)
        plt.colorbar(scatter, label='Writer')
        plt.title(f't-SNE Embedding of {run.name} - Run {run_id}')
        #plt.xlabel('TSNE Component 1')
        #plt.ylabel('TSNE Component 2')
        plt.savefig(os.path.join(folder, f'tsne_{table_key}.svg'), format='svg')
        plt.show()

def plot_history(run_id, folder):
    run = api.run(f"{wandb_entity}/{wandb_project}/{run_id}")
    
    for key in ["Val-mAP", "loss", "lr", "Epoch"]:
        history = run.history(keys=[key])
        if not history.empty:
            plt.figure(figsize=(8, 8))
            plt.plot(history["_step"], history[key], label=key)
            plt.xlabel("Steps")
            plt.legend()
            plt.title(f'{run.name} {key}')
            plt.savefig(os.path.join(folder, f'{key}.svg'), format='svg')
            plt.show()

def generate_latex_table(run_id, folder):
    run = api.run(f"{wandb_entity}/{wandb_project}/{run_id}")
    results_table = fetch_table(run, 'Results', folder)

    if results_table is not None:
        results_table['Pooling'] = results_table['Pooling'].replace({
            "SumPooling-PN0p4-512": "Base",
            "SumPooling-PN0p4-512-reranking": "Reranked"
        })
    
        latex_table = results_table.to_latex(index=False, column_format='|c|c|c|', header=["", "mAP", "Top1"], escape=False)#, lines=True)
    
        with open(os.path.join(folder, 'results_table.tex'), 'w') as f:
            f.write(latex_table)

def generate_latex_file(run_id, folder):
    latex_content = r"""
    \documentclass{article}
    \usepackage{graphicx}
    \usepackage{svg}
    \usepackage{booktabs}
    
    \begin{document}
    
    \section*{t-SNE Plots}
    
    \begin{figure}[ht]
        \centering
        \includesvg[width=\textwidth]{tsne_SumPooling-PN0p4_features_reranked}
        \caption{t-SNE Plot for SumPooling-PN0p4_features_reranked}
    \end{figure}
    
    \begin{figure}[ht]
        \centering
        \includesvg[width=\textwidth]{tsne_SumPooling-PN0p4_features}
        \caption{t-SNE Plot for SumPooling-PN0p4_features}
    \end{figure}
    
    \section*{History Metrics}
    
    \begin{figure}[ht]
        \centering
        \includesvg[width=\textwidth]{Val-mAP}
        \caption{Val-mAP}
    \end{figure}
    
    \begin{figure}[ht]
        \centering
        \includesvg[width=\textwidth]{loss}
        \caption{loss}
    \end{figure}
    
    \begin{figure}[ht]
        \centering
        \includesvg[width=\textwidth]{lr}
        \caption{lr}
    \end{figure}
    
    \begin{figure}[ht]
        \centering
        \includesvg[width=\textwidth]{Epoch}
        \caption{Epoch}
    \end{figure}
    
    \section*{Results Table}
    
    \input{results_table.tex}
    
    \end{document}
        """
    with open(os.path.join(folder, 'report.tex'), 'w') as f:
        f.write(latex_content)

def generate_comprehensive_latex(run_folders):
    latex_content = r"""
    \documentclass{article}
    \usepackage{graphicx}
    \usepackage{svg}
    \usepackage{booktabs}
    
    \begin{document}
    
    """
    for folder in run_folders:
        run_id = os.path.basename(folder).split("_")[0]
        latex_content += f"\\section*{{Run {run_id}}}\n"
        latex_content += f"\\input{{{os.path.join(folder, 'report.tex')}}}\n"

    latex_content += r"""
\end{document}
    """
    with open('comprehensive_report.tex', 'w') as f:
        f.write(latex_content)

if __name__ == '__main__':

    wandb_entity = "mura1i"
    wandb_project = "icadr23"
    
    # Authenticate and set the entity and project
    api = wandb.Api()
    
    tag = "generate-plots"
    run_ids = get_run_ids_by_tag(tag)
    run_folders = []
    
    for run_id in run_ids:
        run = api.run(f"{wandb_entity}/{wandb_project}/{run_id}")
        run_name = run.name.replace(" ", "_")  # Replace spaces with underscores for folder names
        folder_name = f"{run_id}_{run_name}"
        #os.makedirs(folder_name, exist_ok=True)
        #run_folders.append(folder_name)
        basepath = "/cluster/qy41tewa/rl-map/rlmap/icadr23/"
        folder_name = os.path.join(basepath,folder_name)
        
        # Download artifacts
       # download_artifacts(run_id, folder_name)
        
        # t-SNE plots
        for table_key in ["SumPooling-PN0p4_features_reranked", "SumPooling-PN0p4_features"]:
            plot_tsne(run_id, table_key, folder_name)

        # History plots
        plot_history(run_id, folder_name)

        # LaTeX table
        generate_latex_table(run_id, folder_name)

        # Generate LaTeX file
        generate_latex_file(run_id, folder_name)

    # Generate comprehensive LaTeX file
    generate_comprehensive_latex(run_folders)
