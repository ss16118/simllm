# TRAF-94 H100 primitive-identification results

Outcome: **FAIL**.

## Evidence

- Work items: 3840
- Raw rows: 1152000
- Qualified cells: 768
- Fatal guards: `valid_outside_void_scopes`
- Voided cell/timer scopes: 8
- Observation SHA-256: `594769e4d2871f119a05f47dd5f8a60cc03e8bb3475766da2d8e1162f86ecb5d`
- Fit digest: `f4d53fa06fffc0576ce0751bd9af638b447f2ec3f4982e9b7034ac33bfc6fce1`
- Score digest: `e428276b490949b8945b1bbcc3540870168df81926657df8d9d03421b8fbfd19`

## Identification

The fit used `identification_cells_only` and 0 confirmation rows. It contains 1792 process-median anchors. Of 3012 matched contrasts, 656 resolve as separate terms and 2356 remain joint intervals.

## Held-out confirmation

| Timer | Protocol | Placement | Factor | Total | Resolved | Accepted | Rejected | Unresolved | Worst error / bound |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| cuda_event | LL | source_default | active_channels | 24 | 20 | 0 | 20 | 4 | 100.5616 |
| cuda_event | LL | source_default | available_sms | 24 | 1 | 0 | 1 | 23 | 1.4453 |
| cuda_event | LL | source_default | delay_cycles | 24 | 0 | 0 | 0 | 24 | — |
| cuda_event | LL | source_default | ranks | 24 | 13 | 5 | 8 | 11 | 34.0507 |
| cuda_event | LL | source_default | useful_bytes_per_channel | 32 | 29 | 9 | 20 | 3 | 7.5821 |
| cuda_event | LL128 | source_default | active_channels | 24 | 20 | 2 | 18 | 4 | 6.4554 |
| cuda_event | LL128 | source_default | available_sms | 24 | 1 | 0 | 1 | 23 | 3.2502 |
| cuda_event | LL128 | source_default | delay_cycles | 24 | 0 | 0 | 0 | 24 | — |
| cuda_event | LL128 | source_default | ranks | 24 | 14 | 7 | 7 | 10 | 4.8895 |
| cuda_event | LL128 | source_default | useful_bytes_per_channel | 32 | 14 | 2 | 12 | 18 | 6.3052 |
| cuda_event | SIMPLE | buffered | active_channels | 24 | 6 | 4 | 2 | 18 | 8.1559 |
| cuda_event | SIMPLE | buffered | available_sms | 24 | 0 | 0 | 0 | 24 | — |
| cuda_event | SIMPLE | buffered | delay_cycles | 24 | 0 | 0 | 0 | 24 | — |
| cuda_event | SIMPLE | buffered | ranks | 24 | 24 | 19 | 5 | 0 | 1.4558 |
| cuda_event | SIMPLE | buffered | useful_bytes_per_channel | 32 | 2 | 0 | 2 | 30 | 10.6537 |
| cuda_event | SIMPLE | buffered_read | active_channels | 12 | 2 | 0 | 2 | 10 | 10.3827 |
| cuda_event | SIMPLE | buffered_read | available_sms | 12 | 1 | 0 | 1 | 11 | 10.8515 |
| cuda_event | SIMPLE | buffered_read | delay_cycles | 12 | 1 | 0 | 1 | 11 | 8.6235 |
| cuda_event | SIMPLE | buffered_read | useful_bytes_per_channel | 16 | 1 | 0 | 1 | 15 | 8.0158 |
| host_wall | LL | source_default | active_channels | 24 | 20 | 0 | 20 | 4 | 64.7566 |
| host_wall | LL | source_default | available_sms | 24 | 1 | 0 | 1 | 23 | 3.3894 |
| host_wall | LL | source_default | delay_cycles | 24 | 0 | 0 | 0 | 24 | — |
| host_wall | LL | source_default | ranks | 24 | 13 | 4 | 9 | 11 | 22.7244 |
| host_wall | LL | source_default | useful_bytes_per_channel | 32 | 29 | 3 | 26 | 3 | 9.9629 |
| host_wall | LL128 | source_default | active_channels | 24 | 20 | 2 | 18 | 4 | 6.4150 |
| host_wall | LL128 | source_default | available_sms | 24 | 0 | 0 | 0 | 24 | — |
| host_wall | LL128 | source_default | delay_cycles | 24 | 0 | 0 | 0 | 24 | — |
| host_wall | LL128 | source_default | ranks | 24 | 15 | 8 | 7 | 9 | 4.9961 |
| host_wall | LL128 | source_default | useful_bytes_per_channel | 32 | 15 | 2 | 13 | 17 | 9.6319 |
| host_wall | SIMPLE | buffered | active_channels | 24 | 5 | 2 | 3 | 19 | 8.1477 |
| host_wall | SIMPLE | buffered | available_sms | 24 | 0 | 0 | 0 | 24 | — |
| host_wall | SIMPLE | buffered | delay_cycles | 24 | 0 | 0 | 0 | 24 | — |
| host_wall | SIMPLE | buffered | ranks | 24 | 24 | 2 | 22 | 0 | 6.9937 |
| host_wall | SIMPLE | buffered | useful_bytes_per_channel | 32 | 2 | 0 | 2 | 30 | 10.4910 |
| host_wall | SIMPLE | buffered_read | active_channels | 12 | 2 | 0 | 2 | 10 | 9.2610 |
| host_wall | SIMPLE | buffered_read | available_sms | 12 | 1 | 0 | 1 | 11 | 10.0213 |
| host_wall | SIMPLE | buffered_read | delay_cycles | 12 | 1 | 0 | 1 | 11 | 8.5827 |
| host_wall | SIMPLE | buffered_read | useful_bytes_per_channel | 16 | 1 | 0 | 1 | 15 | 7.7688 |
| same_device_interval | LL | source_default | active_channels | 24 | 16 | 0 | 16 | 8 | 129.9142 |
| same_device_interval | LL | source_default | available_sms | 24 | 0 | 0 | 0 | 24 | — |
| same_device_interval | LL | source_default | delay_cycles | 24 | 0 | 0 | 0 | 24 | — |
| same_device_interval | LL | source_default | ranks | 24 | 7 | 3 | 4 | 17 | 15.3139 |
| same_device_interval | LL | source_default | useful_bytes_per_channel | 32 | 29 | 9 | 20 | 3 | 7.7781 |
| same_device_interval | LL128 | source_default | active_channels | 24 | 20 | 0 | 20 | 4 | 9.1043 |
| same_device_interval | LL128 | source_default | available_sms | 24 | 1 | 0 | 1 | 23 | 5.2273 |
| same_device_interval | LL128 | source_default | delay_cycles | 24 | 1 | 0 | 1 | 23 | 4.5077 |
| same_device_interval | LL128 | source_default | ranks | 24 | 13 | 5 | 8 | 11 | 7.1515 |
| same_device_interval | LL128 | source_default | useful_bytes_per_channel | 32 | 16 | 3 | 13 | 16 | 5.3092 |
| same_device_interval | SIMPLE | buffered | active_channels | 24 | 7 | 0 | 7 | 17 | 285.0444 |
| same_device_interval | SIMPLE | buffered | available_sms | 24 | 1 | 0 | 1 | 23 | 3.3714 |
| same_device_interval | SIMPLE | buffered | delay_cycles | 24 | 0 | 0 | 0 | 24 | — |
| same_device_interval | SIMPLE | buffered | ranks | 24 | 24 | 6 | 18 | 0 | 32.3997 |
| same_device_interval | SIMPLE | buffered | useful_bytes_per_channel | 32 | 0 | 0 | 0 | 32 | — |
| same_device_interval | SIMPLE | buffered_read | active_channels | 12 | 8 | 0 | 8 | 4 | 260.4906 |
| same_device_interval | SIMPLE | buffered_read | available_sms | 12 | 0 | 0 | 0 | 12 | — |
| same_device_interval | SIMPLE | buffered_read | delay_cycles | 12 | 0 | 0 | 0 | 12 | — |
| same_device_interval | SIMPLE | buffered_read | useful_bytes_per_channel | 16 | 0 | 0 | 0 | 16 | — |

Resolved interventions: 441; accepted: 97; rejected: 344.

## Claim boundary

TRAF-94 identifies primitive component response on the qualified H100 scope. TRAF-43 retains full collective accuracy and uncertainty; TRAF-54 retains integration into finite channel/GPU resources.

The adjacent JSON result is authoritative for exact component surfaces, grouped contrasts, void scopes, identities, and input hashes.
