# Verbatim copies of utility modules from the original ChromBERT repo.
# Do not edit these files — adapters live in flashchrombert.data, not here.
#
# Submodules have disjoint heavy-dep trees, so we do NOT eagerly import them
# here. Pull in only what you need:
#     from flashchrombert.legacy import css_utility    # needs matplotlib/tslearn/networkx/…
#     from flashchrombert.legacy import motif_utils    # needs scipy/statsmodels/biopython/ahocorasick
#     from flashchrombert.legacy import find_motifs    # CLI: python -m flashchrombert.legacy.find_motifs
# Install the full legacy extra via:  pip install -e .[legacy]
