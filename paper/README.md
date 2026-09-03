# Decodable Is Not Direction-Specific

LaTeX source and reproducible figures for the evidence-locked ICLR draft.

## Build

From this directory:

```bash
make assets
make
make check
```

`make assets` reads the four accepted immutable run summaries under
`/home/qingchan/data/concept-flow`, validates the adjudicated headline values, writes the compact
`data/accepted_results.json` snapshot, and regenerates Figures 1--3 and Tables 1--2. The run identifiers
and primary artifact paths remain embedded in that snapshot.

The manuscript uses the anonymous ICLR style. The tracked PDF is built with Tectonic. Raw experiment
outputs, model weights, and image data remain in the external run store.
