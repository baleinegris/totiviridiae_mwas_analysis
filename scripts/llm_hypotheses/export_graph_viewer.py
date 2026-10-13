"""Export Leiden clusters of related findings to the DittoGraphVisualization viewer format.

Reads the clusters from leiden_clustering.ipynb and the related pairs from link_findings.ipynb, and writes
clusters.csv {name, cluster_id, pvalue, cohens_d} and edges.csv {name1, name2, score} into a subfolder of the viewer's
public/data/, then registers the dataset in its index.json so it shows up in the dropdown.

The viewer lays out the graph in the browser, so only the largest clusters are exported, up to --max-nodes.

usage: python export_graph_viewer.py [--name NAME] [--min-cluster-size N] [--max-nodes N] [--viewer-data DIR]
"""

import argparse
import json
import re
from pathlib import Path

import polars as pl

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
DEFAULT_VIEWER_DATA = Path.home() / "Documents" / "DittoGraphVisualization" / "viewer" / "public" / "data"

# longest attribute value kept in a node label, the longest values run to over 1000 characters
MAX_VALUE_CHARS = 60


def clean(text):
    # the viewer splits CSV lines on commas without handling quotes, and puts names into innerHTML and a
    # double-quoted data attribute, so commas, quotes and angle brackets would break parsing or rendering
    text = re.sub(r"\s+", " ", "null" if text is None else text).strip()
    text = text.replace(",", ";").replace('"', "'").replace("<", "‹").replace(">", "›")
    if len(text) > MAX_VALUE_CHARS:
        text = text[: MAX_VALUE_CHARS - 1] + "…"
    return text


def node_label(row):
    attributes = "; ".join(
        f"{clean(name.removesuffix('_attrib'))}={clean(value)}" for name, value in zip(row["col_name"], row["col_value"])
    )
    # the row index keeps names unique and maps a node back to its finding
    return f"[{row['row_idx']}] {row['bioproject']}: {attributes}"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", default="totiviridiae-hypotheses", help="dataset name shown in the viewer dropdown")
    parser.add_argument("--min-cluster-size", type=int, default=2, help="drop clusters smaller than this")
    parser.add_argument("--max-nodes", type=int, default=3000, help="export the largest clusters up to this many nodes")
    parser.add_argument("--viewer-data", type=Path, default=DEFAULT_VIEWER_DATA, help="the viewer's public/data folder")
    args = parser.parse_args()

    # row i of the clusters table is finding i, which is what row_1 and row_2 index into
    findings = pl.read_parquet(DATA_DIR / "hypotheses_clusters.parquet").with_row_index("row_idx")
    related = pl.read_parquet(DATA_DIR / "hypotheses_related_findings.parquet")

    # pick the largest clusters until the node budget is used up, ties broken by cluster id so the export is stable
    sizes = (
        findings.group_by("cluster_id")
        .len()
        .filter(pl.col("len") >= args.min_cluster_size)
        .sort(["len", "cluster_id"], descending=[True, False])
        .filter(pl.col("len").cum_sum() <= args.max_nodes)
        # renumber so the largest cluster is 0, the viewer colours clusters by id
        .with_row_index("new_cluster_id")
    )

    nodes = (
        findings.join(sizes.select("cluster_id", "new_cluster_id"), on="cluster_id")
        .with_columns(
            pl.struct(["row_idx", "bioproject", "col_name", "col_value"])
            .map_elements(node_label, return_dtype=pl.String)
            .alias("name")
        )
        .select("row_idx", "name", pl.col("new_cluster_id").alias("cluster_id"), "pvalue", "cohens_d")
        .sort("cluster_id", "row_idx")
    )

    # keep edges between exported findings, deduplicated since the graph is undirected, related pairs are unweighted
    names = nodes.select("row_idx", "name")
    edges = (
        related.filter(pl.col("row_1") != pl.col("row_2"))
        .select(
            pl.min_horizontal("row_1", "row_2").cast(pl.UInt32).alias("source"),
            pl.max_horizontal("row_1", "row_2").cast(pl.UInt32).alias("target"),
        )
        .unique()
        .join(names.rename({"row_idx": "source", "name": "name1"}), on="source")
        .join(names.rename({"row_idx": "target", "name": "name2"}), on="target")
        .sort("source", "target")
        .select("name1", "name2", pl.lit(1.0).alias("score"))
    )

    out_dir = args.viewer_data / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    nodes.select("name", "cluster_id", "pvalue", "cohens_d").write_csv(out_dir / "clusters.csv", quote_style="never")
    edges.write_csv(out_dir / "edges.csv", quote_style="never")

    index_path = args.viewer_data / "index.json"
    datasets = json.loads(index_path.read_text(encoding="utf-8"))
    if args.name not in datasets:
        datasets.append(args.name)
        index_path.write_text(json.dumps(datasets), encoding="utf-8")

    print(f"wrote {nodes.height} findings in {sizes.height} clusters and {edges.height} edges to {out_dir}")


if __name__ == "__main__":
    main()
