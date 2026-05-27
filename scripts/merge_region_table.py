#!/usr/bin/env python3
"""Merge raw baseline stats with existing on-disk and resident breakdown data
to produce the final per-region compression ratio table."""

import sys

# === Raw baseline data (from raw_baseline_stats runs) ===
raw_data = {
    "leworldmodel_full": {
        "ts": 27755112, "dur": 27755112, "args": 182029625,
        "names": 146286155, "tree_refs": 55510224, "ids_pids_streams": 83265336,
        "total_events": 3469389,
    },
    "qwen3_full": {
        "ts": 270508592, "dur": 270508592, "args": 1064007965,
        "names": 1168136564, "tree_refs": 541017184, "ids_pids_streams": 811525776,
        "total_events": 33813574,
    },
    "unifolm_full": {
        "ts": 641784568, "dur": 641784568, "args": 4967110923,
        "names": 4829430919, "tree_refs": 1283569136, "ids_pids_streams": 1925353704,
        "total_events": 80223071,
    },
    "llama_full": {
        "ts": 2410304928, "dur": 2410304928, "args": 11645569308,
        "names": 15276406013, "tree_refs": 4820609856, "ids_pids_streams": 7230914784,
        "total_events": 301288116,
    },
}

# === PADoC on-disk data (from on_disk_breakdown.txt) ===
# Mapping: region -> (msgpack_bytes, zstd_bytes)
padoc_disk = {
    "leworldmodel_full": {
        "ts": (16292794, 4713353),
        "dur": (3414947, 339083),
        "args": (53189815, 20501632),
        "names": (665380 + 926328, 57946 + 53837),  # template_headers + name_nums
        "tree_refs": (77928587 + 188215449, 12954184 + 13573761),  # rank_node_tree + node_soft_links
        "ids_pids_streams": (270878, 45336),
    },
    "qwen3_full": {
        "ts": (168976140, 125503072),
        "dur": (35295054, 14482656),
        "args": (286246715, 44784875),
        "names": (675808 + 16427358, 47646 + 337909),
        "tree_refs": (709972834 + 1296194922, 100147734 + 100638858),
        "ids_pids_streams": (16948693, 28582),
    },
    "unifolm_full": {
        "ts": (400762973, 144343518),
        "dur": (72828998, 6673602),
        "args": (1331607162, 352396895),
        "names": (2778337 + 30086655, 201646 + 934417),
        "tree_refs": (1198922883 + 2398214600, 258250526 + 264151919),
        "ids_pids_streams": (137926931, 3840454),
    },
    "llama_full": {
        "ts": (1505735597, 1073867885),
        "dur": (259492391, 101498433),
        "args": (2037114675, 295247430),
        "names": (85398 + 401632500, 8105 + 2715745),
        "tree_refs": (5301133265 + 8548916485, 946110875 + 925145515),
        "ids_pids_streams": (573876467, 153426779),
    },
}

# === PADoC resident memory data (from on_disk_breakdown.txt) ===
# Mapping: region -> resident_bytes
padoc_resident = {
    "leworldmodel_full": {
        "ts": 14159504 + 121740,  # cpu_ts + gpu_ts
        "dur": 13560932 + 121672,  # cpu_dur + gpu_dur
        "args": 48544532 + 5501763,  # arg_vec_storage + arg_payload
        "names": 1891996 + 14512 + 393242,  # name_num_vecs + name_num_payload + string_payload_other
        "tree_refs": 210585576 + 9135660,  # node_vec_storage + node_u32_vec_storage
        "ids_pids_streams": 218544 + 121728 + 365 + 175776,  # cpu_id + gpu_pid + gpu_ph + gpu_stream
    },
    "qwen3_full": {
        "ts": 151223204 + 11900360,
        "dur": 113474464 + 11900360,
        "args": 451094036 + 157549679,
        "names": 27203016 + 274548 + 324433,
        "tree_refs": 1607469928 + 119701780,
        "ids_pids_streams": 0 + 11899872 + 420 + 47137632,
    },
    "unifolm_full": {
        "ts": 377116980 + 38867488,
        "dur": 305253024 + 38677460,
        "args": 1931088124 + 327554009,
        "names": 65649616 + 54800 + 1655708,
        "tree_refs": 3375081336 + 316241096,
        "ids_pids_streams": 71303168 + 38867488 + 1320 + 331180485,
    },
    "llama_full": {
        "ts": 1617974572 + 171115712,
        "dur": 1047549204 + 171115712,
        "args": 3808381824 + 100453388,
        "names": 2502784904 + 1063 + 63100,
        "tree_refs": 11965270824 + 1138670880,
        "ids_pids_streams": 570425344 + 171115712 + 490 + 468040289,
    },
}

# === Output TSV ===
regions = ["ts", "dur", "args", "names", "tree_refs", "ids_pids_streams"]
datasets = ["leworldmodel_full", "qwen3_full", "unifolm_full", "llama_full"]

header = "dataset\tregion\traw_bytes\tpadoc_msgpack_bytes\tpadoc_zstd_bytes\tpadoc_resident_bytes\tratio_disk\tratio_memory"
print(header)

for ds in datasets:
    for region in regions:
        raw = raw_data[ds][region]
        msgpack, zstd = padoc_disk[ds][region]
        resident = padoc_resident[ds][region]
        ratio_disk = raw / zstd if zstd > 0 else 0
        ratio_mem = raw / resident if resident > 0 else 0
        print(f"{ds}\t{region}\t{raw}\t{msgpack}\t{zstd}\t{resident}\t{ratio_disk:.2f}\t{ratio_mem:.2f}")
