# Decodable Is Not Direction-Specific

LaTeX source and reproducible figures for the evidence-locked ICLR draft.

## Build

From the repository root, activate the fixed project environment and enter the paper directory:

```bash
source scripts/server/activate_env.sh
cd paper
make assets
make
make check
```

The first Tectonic build populates its user cache from the network. Later builds reuse the cached
format, packages, and fonts. Bare `make` builds `main.pdf`; `make check` reports the page count and
layout-warning counts for that PDF.

`make assets` reads the four accepted immutable run summaries under
`/home/qingchan/data/concept-flow`, validates the adjudicated headline values, writes the compact
`data/accepted_results.json` snapshot, and regenerates Figures 1--3 and Tables 1--2. The run identifiers
and primary artifact paths remain embedded in that snapshot.

The manuscript uses the anonymous ICLR style. The tracked PDF is built with Tectonic. Raw experiment
outputs, model weights, and image data remain in the external run store.
