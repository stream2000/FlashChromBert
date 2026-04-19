import os
import glob
import pandas as pd
from collections import defaultdict

def calc_tds(motif: str) -> int:
    if not motif or pd.isna(motif):
        return 0
    motif = str(motif)
    tds = 0
    curr = motif[0]
    for char in motif[1:]:
        if char != curr:
            tds += 1
            curr = char
    return tds

def main():
    base_dir = "logs/motif"
    csv_files = glob.glob(f"{base_dir}/**/result/init_df.csv", recursive=True)
    
    results = []
    
    for f in csv_files:
        # File path like: logs/motif/cls_not_n_rpkm0_wg/result/init_df.csv
        parts = f.split(os.sep)
        run_name = parts[2]  # e.g., cls_not_n_rpkm0_wg
        
        # parse the task, dataset, and model type
        if run_name.endswith("_wg"):
            model_type = "Pretrained (WG)"
            dataset = run_name[:-3]
        elif run_name.endswith("_random"):
            model_type = "Random-Init"
            dataset = run_name[:-7]
        else:
            model_type = "Unknown"
            dataset = run_name
            
        try:
            df = pd.read_csv(f)
            # Find the motif column. It might be named 'motif' or similar.
            # Assuming 'motif' based on typical output
            if 'motif' not in df.columns:
                print(f"Warning: 'motif' column not found in {f}")
                continue
                
            num_motifs = len(df)
            if num_motifs > 0:
                avg_len = df['motif'].apply(lambda x: len(str(x))).mean()
                df['tds'] = df['motif'].apply(calc_tds)
                # tds enrich ratio: % of motifs with TDS >= 1
                tds_enrich_ratio = (df['tds'] >= 1).mean() * 100
                avg_tds = df['tds'].mean()
            else:
                avg_len = 0.0
                tds_enrich_ratio = 0.0
                avg_tds = 0.0
                
            results.append({
                "Dataset": dataset,
                "Model": model_type,
                "Motif Count": num_motifs,
                "Avg Length": round(avg_len, 2),
                "Avg TDS": round(avg_tds, 2),
                "TDS >= 1 (%)": round(tds_enrich_ratio, 2)
            })
        except Exception as e:
            print(f"Error processing {f}: {e}")
            
    if not results:
        print("No valid init_df.csv files processed.")
        return
        
    df_res = pd.DataFrame(results)
    df_res = df_res.sort_values(by=["Dataset", "Model"], ascending=[True, False])
    
    print("=" * 90)
    print(f"{'Dataset':<30} | {'Model':<16} | {'Count':<6} | {'Avg Len':<8} | {'Avg TDS':<8} | {'TDS>=1 (%)':<10}")
    print("-" * 90)
    for _, row in df_res.iterrows():
        print(f"{row['Dataset']:<30} | {row['Model']:<16} | {row['Motif Count']:<6} | {row['Avg Length']:<8} | {row['Avg TDS']:<8} | {row['TDS >= 1 (%)']:<10}")
    print("=" * 90)
    
if __name__ == "__main__":
    main()
