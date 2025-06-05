import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def extract_confidence_data(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)

    records = []
    for task in data.get("answer", []):
        confidence = task["result"].get("confidence_level", "unknown")
        agreement = task["result"].get("agreement_level", "unknown")  # opcional si lo defines en el output
        records.append({
            "task_id": task["task_id"],
            "task_name": task["task_name"],
            "confidence_level": confidence,
            "agreement_level": agreement
        })
    return pd.DataFrame(records)

def plot_heatmap(df, output_file="confidence_heatmap.png"):
    # Convertimos niveles a valores numéricos
    mapping = {"low": 0, "medium": 1, "high": 2, "unknown": -1}
    df["score"] = df["confidence_level"].map(mapping)

    # Creamos una matriz de calor
    pivot = df.pivot(index="task_name", columns="task_id", values="score")
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, len(df) * 0.5 + 2))
    ax = sns.heatmap(pivot, annot=True, cmap="YlGnBu", cbar_kws={"label": "Confidence Level"})
    ax.set_title("Confidence Heatmap por Tarea")
    plt.tight_layout()
    plt.savefig(output_file)
    print(f"Heatmap guardado en: {output_file}")
    plt.show()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Visualizar Heatmap de Confianza por Tarea")
    parser.add_argument("json_file", type=str, help="Ruta al archivo JSON con los resultados")
    args = parser.parse_args()

    df = extract_confidence_data(args.json_file)
    if df.empty:
        print("No hay datos de confianza para visualizar.")
    else:
        plot_heatmap(df)