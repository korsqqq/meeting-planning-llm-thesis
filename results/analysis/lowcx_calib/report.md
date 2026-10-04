# Lower-complexity calibration — C1 vs C3

**exploratory development calibration; descriptive only. No hypothesis test, no threshold, no cap selection, no gate.** Twelve instances per size at each cap: description and ordering, not inference.

216 runs over 36 frozen instances, model Qwen/Qwen3-32B-AWQ. Completeness audit: 216/216 verified, passed = True.

Every instance has oracle optimum 3, so satisfaction takes the same four values {0, 1/3, 2/3, 1} at every size and no difference below can come from the denominator.

## Satisfaction — the primary metric

Sign: `delta = C1 - C3`, so a **negative** delta means the hierarchy is ahead.

| n | cap | C1 | C3 | Δ (C1−C3) | C1 ahead | C3 ahead | tied |
|---|---|---|---|---|---|---|---|
| 4 | 8000 | 0.000 | 0.000 | +0.000 | 0 | 0 | 12 |
| 4 | 16000 | 0.083 | 0.194 | -0.111 | 1 | 6 | 5 |
| 4 | 32000 | 0.500 | 0.722 | -0.222 | 3 | 6 | 3 |
| 5 | 8000 | 0.000 | 0.000 | +0.000 | 0 | 0 | 12 |
| 5 | 16000 | 0.000 | 0.139 | -0.139 | 0 | 5 | 7 |
| 5 | 32000 | 0.278 | 0.583 | -0.306 | 2 | 7 | 3 |
| 6 | 8000 | 0.000 | 0.000 | +0.000 | 0 | 0 | 12 |
| 6 | 16000 | 0.000 | 0.056 | -0.056 | 0 | 2 | 10 |
| 6 | 32000 | 0.000 | 0.222 | -0.222 | 0 | 5 | 7 |

### Pooled over sizes, by cap

| cap | C1 | C3 | Δ (C1−C3) | C1 ahead | C3 ahead | tied |
|---|---|---|---|---|---|---|
| 8000 | 0.000 | 0.000 | +0.000 | 0 | 0 | 36 |
| 16000 | 0.028 | 0.130 | -0.102 | 1 | 13 | 22 |
| 32000 | 0.259 | 0.509 | -0.250 | 5 | 18 | 13 |

### Pooled over caps, by size

*`n` is a generator parameter, not a complexity level. It is varied because `H <= n - O` bounds the higher-order interaction; `n` and the attainable `H` are tied by that inequality and their separate effects are not identified here.*

| n | C1 | C3 | Δ (C1−C3) | C1 ahead | C3 ahead | tied |
|---|---|---|---|---|---|---|
| 4 | 0.194 | 0.306 | -0.111 | 4 | 12 | 20 |
| 5 | 0.093 | 0.241 | -0.148 | 2 | 12 | 22 |
| 6 | 0.000 | 0.093 | -0.093 | 0 | 7 | 29 |

## Budget — can a poor result be starvation rather than difficulty?

C3 splits its working budget; each worker gets `floor(3/8 * (cap - 256))`:

| cap | C3 worker quota | two workers |
|---|---|---|
| 8000 | 2904 | 5808 |
| 16000 | 5904 | 11808 |
| 32000 | 11904 | 23808 |

| n | cap | condition | mean tokens | cap used | calls | termination |
|---|---|---|---|---|---|---|
| 4 | 8000 | c1_react | 7790 | 0.97 | 8.1 | {'aborted': 10, 'budget': 2} |
| 4 | 8000 | c3_mas | 5689 | 0.71 | 7.1 | {'aborted': 5, 'agent_finish': 7} |
| 4 | 16000 | c1_react | 15364 | 0.96 | 10.8 | {'aborted': 8, 'agent_finish': 1, 'budget': 3} |
| 4 | 16000 | c3_mas | 11858 | 0.74 | 12.7 | {'aborted': 2, 'agent_finish': 10} |
| 4 | 32000 | c1_react | 25581 | 0.80 | 13.0 | {'aborted': 4, 'agent_finish': 7, 'budget': 1} |
| 4 | 32000 | c3_mas | 22546 | 0.70 | 16.5 | {'agent_finish': 12} |
| 5 | 8000 | c1_react | 7708 | 0.96 | 8.2 | {'aborted': 8, 'budget': 4} |
| 5 | 8000 | c3_mas | 5706 | 0.71 | 7.1 | {'aborted': 3, 'agent_finish': 9} |
| 5 | 16000 | c1_react | 15821 | 0.99 | 10.7 | {'aborted': 11, 'budget': 1} |
| 5 | 16000 | c3_mas | 11900 | 0.74 | 13.1 | {'aborted': 6, 'agent_finish': 6} |
| 5 | 32000 | c1_react | 29139 | 0.91 | 13.8 | {'aborted': 7, 'agent_finish': 3, 'budget': 2} |
| 5 | 32000 | c3_mas | 22540 | 0.70 | 17.1 | {'aborted': 1, 'agent_finish': 11} |
| 6 | 8000 | c1_react | 7757 | 0.97 | 7.8 | {'aborted': 8, 'budget': 4} |
| 6 | 8000 | c3_mas | 5821 | 0.73 | 7.0 | {'aborted': 5, 'agent_finish': 7} |
| 6 | 16000 | c1_react | 15618 | 0.98 | 10.5 | {'aborted': 5, 'budget': 7} |
| 6 | 16000 | c3_mas | 11884 | 0.74 | 12.8 | {'aborted': 8, 'agent_finish': 4} |
| 6 | 32000 | c1_react | 30972 | 0.97 | 14.7 | {'aborted': 9, 'agent_finish': 2, 'budget': 1} |
| 6 | 32000 | c3_mas | 22950 | 0.72 | 17.4 | {'aborted': 4, 'agent_finish': 8} |

## Validated non-empty rate — secondary, diagnostic

A structural precondition for scoring at all. **Never** described as performance or accuracy.

| n | cap | C1 | C3 |
|---|---|---|---|
| 4 | 8000 | 0/12 | 0/12 |
| 4 | 16000 | 1/12 | 6/12 |
| 4 | 32000 | 7/12 | 12/12 |
| 5 | 8000 | 0/12 | 0/12 |
| 5 | 16000 | 0/12 | 5/12 |
| 5 | 32000 | 4/12 | 11/12 |
| 6 | 8000 | 0/12 | 0/12 |
| 6 | 16000 | 0/12 | 2/12 |
| 6 | 32000 | 0/12 | 5/12 |

---

*No threshold is introduced, no cap is selected and no gate is applied. Whether any of this enters the held-out design is a separate dated decision.*
