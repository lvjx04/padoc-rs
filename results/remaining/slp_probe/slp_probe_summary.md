# SLP Timestamp/Duration Probe

This is an offline probe. It does not change the PADOC artifact format.

## Settings

| Setting | Value |
|---|---:|
| min column length | 1024 |
| max columns per dataset | 128 |
| max sampled values per column | 1000000 |
| max segment length | 65536 |
| fixed-point Q | 32 |
| zstd level | 3 |

## Aggregate Results

| Dataset | Columns | Values | Truncated cols | Best SLP mem / i64 mem | Best SLP vs i64 | Best SLP vs i32 | Accepted cols | Beat i32 cols | Best SLP zstd / i64 zstd | Encode secs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `leworldmodel_full` | 128 | 3906828 | 0 | 4427773 / 31254624 | 7.06x | 3.53x | 128 | 128 | 1850886 / 2747165 | 0.153 |
| `llama_full` | 128 | 93249920 | 62 | 188063448 / 745999360 | 3.97x | 1.98x | 126 | 126 | 163066859 / 202943034 | 4.049 |
| `qwen3_full` | 128 | 38097496 | 12 | 75162780 / 304779968 | 4.05x | 2.03x | 118 | 118 | 60561765 / 85796369 | 1.604 |
| `unifolm_full` | 128 | 58928428 | 21 | 85977084 / 471427424 | 5.48x | 2.74x | 116 | 116 | 30975598 / 43438411 | 2.321 |

Acceptance criterion for the prototype is `best_slp_ratio_vs_i64 > 2.0`.
A separate `beats_i32` flag records whether the SLP result is also smaller than the current i32-style memory baseline.
