import pandas as pd
from flashchrombert.legacy import css_utility

tasks = [
    ('ft_cls_wg (RPKM 0 vs >50)', 'logs/motif/ft_cls_wg/result/init_df.csv', 5),
    ('cls_not_n_rpkm50_wg (High Expr vs Others)', 'logs/motif/cls_not_n_rpkm50_wg/result/init_df.csv', 5),
    ('cls_rpkm0_n_rpkm20_wg (No Expr vs Low Expr)', 'logs/motif/cls_rpkm0_n_rpkm20_wg/result/init_df.csv', 5)
]

for name, path, n_clusters in tasks:
    print(f"\n### {name}")
    try:
        clustered = css_utility.motif_init2class(input_path=path, categorical=True, n_clusters=n_clusters)
        for _, row in clustered.iterrows():
            cluster_id = row['Cluster']
            # Remove the first letter artifact which comes from the Entry index mapping
            motifs = [m[1:] if len(m) > 1 else m for m in row['LetterSequence']]
            # Remove duplicates if any after stripping
            motifs = list(set(motifs))
            motifs_str = ', '.join([f"`{m}`" for m in motifs])
            print(f"- **Cluster {cluster_id} ({len(motifs)} motifs)**: {motifs_str}")
    except Exception as e:
        print(f"Error processing {name}: {e}")
