# Verbatim copy of chrombert_utils from the original ChromBERT repo.
# Do not edit css_utility.py — adapters go in flashchrombert.data, not here.
#
# Heavy deps (matplotlib, seaborn, networkx, tslearn, umap-learn, biopython,
# logomaker, wordcloud, scipy, pybedtools, …) are only pulled in when you
# actually import this package. Install them via the `legacy` extra:
#     pip install -e .[legacy]
from .css_utility import *  # noqa: F401,F403
