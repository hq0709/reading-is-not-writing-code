"""cf-transfer-v1 executor implementation (Mayo Clinic, rohpc cluster).

This package implements the external replication protocol specified in
docs/external-replication/{README.md,protocol.json,return-format.md}. It is written as a
portable adapter layer: the science (loci, probes, directions, doses, templates, scoring and
return artefacts) is fixed by the protocol; machine paths, scheduling and batching are local.
"""
PROTOCOL_ID = "cf-transfer-v1"
