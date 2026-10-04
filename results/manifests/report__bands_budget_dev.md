# Band manifest — budget_dev, 20 per band

Selected from a structural pool by fixed rule. No agent, plan, score or token count entered the selection.

* subset hash `911e4864d6fb8ea2a78ff465f8e55a286d3c48afc1b8403d5e63f231bf96fd91`
* binning hash `8e671cd13dbb262b638048eff58241226f09d5804373534789915bc302ba0dff`
* pool `results/calibration/budget_dev` at commit `4b1c1977c58f4f96782797d05d9ab28c8e7a516e`, 9600 candidates, seeds 30000–30099
* shared optimum histogram `{'3': 12, '4': 8}` (scaled reference `{'3': 13, '4': 7}`, used the scaled reference: False)


## Composition

| band | level | conflict pairs | D | n | optimum | tightness | travel structures |
|---|---|---|---|---|---|---|---|
| low | easy | 0–1 | 0.000–0.036 | 20 | {'3': 12, '4': 8} | [0.6, 0.8] | clustered, line, random, uniform |
| medium | medium | 7–7 | 0.250–0.250 | 20 | {'3': 12, '4': 8} | [0.8, 0.9, 1.0] | clustered, line, random, uniform |
| high | hard | 13–16 | 0.464–0.571 | 20 | {'3': 12, '4': 8} | [0.8, 0.9, 1.0] | clustered, line, random, uniform |

## Instances

| band | instance_id | seed | k | D | O | tightness | overlap | travel |
|---|---|---|---|---|---|---|---|---|
| low | `random-n8-t80-o100-s30000` | 30000 | 1 | 0.036 | 3 | 0.8 | 1.0 | random |
| low | `clustered-n8-t60-o100-s30001` | 30001 | 0 | 0.000 | 4 | 0.6 | 1.0 | clustered |
| low | `random-n8-t80-o80-s30002` | 30002 | 0 | 0.000 | 3 | 0.8 | 0.8 | random |
| low | `clustered-n8-t80-o100-s30003` | 30003 | 1 | 0.036 | 3 | 0.8 | 1.0 | clustered |
| low | `clustered-n8-t60-o100-s30004` | 30004 | 0 | 0.000 | 4 | 0.6 | 1.0 | clustered |
| low | `line-n8-t80-o100-s30005` | 30005 | 0 | 0.000 | 3 | 0.8 | 1.0 | line |
| low | `clustered-n8-t60-o100-s30006` | 30006 | 0 | 0.000 | 4 | 0.6 | 1.0 | clustered |
| low | `uniform-n8-t60-o100-s30007` | 30007 | 0 | 0.000 | 4 | 0.6 | 1.0 | uniform |
| low | `uniform-n8-t60-o80-s30008` | 30008 | 0 | 0.000 | 4 | 0.6 | 0.8 | uniform |
| low | `clustered-n8-t60-o100-s30009` | 30009 | 0 | 0.000 | 4 | 0.6 | 1.0 | clustered |
| low | `clustered-n8-t80-o80-s30010` | 30010 | 1 | 0.036 | 3 | 0.8 | 0.8 | clustered |
| low | `clustered-n8-t60-o100-s30011` | 30011 | 0 | 0.000 | 4 | 0.6 | 1.0 | clustered |
| low | `clustered-n8-t60-o100-s30012` | 30012 | 0 | 0.000 | 4 | 0.6 | 1.0 | clustered |
| low | `line-n8-t80-o100-s30013` | 30013 | 1 | 0.036 | 3 | 0.8 | 1.0 | line |
| low | `random-n8-t80-o80-s30014` | 30014 | 1 | 0.036 | 3 | 0.8 | 0.8 | random |
| low | `uniform-n8-t80-o80-s30016` | 30016 | 1 | 0.036 | 3 | 0.8 | 0.8 | uniform |
| low | `clustered-n8-t80-o80-s30019` | 30019 | 0 | 0.000 | 3 | 0.8 | 0.8 | clustered |
| low | `clustered-n8-t80-o100-s30022` | 30022 | 1 | 0.036 | 3 | 0.8 | 1.0 | clustered |
| low | `random-n8-t80-o100-s30023` | 30023 | 0 | 0.000 | 3 | 0.8 | 1.0 | random |
| low | `clustered-n8-t80-o80-s30025` | 30025 | 1 | 0.036 | 3 | 0.8 | 0.8 | clustered |
| medium | `uniform-n8-t90-o50-s30036` | 30036 | 7 | 0.250 | 4 | 0.9 | 0.5 | uniform |
| medium | `line-n8-t90-o80-s30039` | 30039 | 7 | 0.250 | 3 | 0.9 | 0.8 | line |
| medium | `line-n8-t80-o100-s30040` | 30040 | 7 | 0.250 | 3 | 0.8 | 1.0 | line |
| medium | `clustered-n8-t90-o20-s30041` | 30041 | 7 | 0.250 | 4 | 0.9 | 0.2 | clustered |
| medium | `clustered-n8-t80-o100-s30042` | 30042 | 7 | 0.250 | 3 | 0.8 | 1.0 | clustered |
| medium | `clustered-n8-t100-o20-s30043` | 30043 | 7 | 0.250 | 4 | 1.0 | 0.2 | clustered |
| medium | `random-n8-t100-o20-s30044` | 30044 | 7 | 0.250 | 4 | 1.0 | 0.2 | random |
| medium | `clustered-n8-t90-o50-s30045` | 30045 | 7 | 0.250 | 4 | 0.9 | 0.5 | clustered |
| medium | `line-n8-t80-o100-s30047` | 30047 | 7 | 0.250 | 3 | 0.8 | 1.0 | line |
| medium | `clustered-n8-t80-o100-s30048` | 30048 | 7 | 0.250 | 3 | 0.8 | 1.0 | clustered |
| medium | `line-n8-t80-o100-s30049` | 30049 | 7 | 0.250 | 3 | 0.8 | 1.0 | line |
| medium | `random-n8-t100-o20-s30054` | 30054 | 7 | 0.250 | 3 | 1.0 | 0.2 | random |
| medium | `line-n8-t100-o20-s30055` | 30055 | 7 | 0.250 | 3 | 1.0 | 0.2 | line |
| medium | `random-n8-t90-o20-s30056` | 30056 | 7 | 0.250 | 4 | 0.9 | 0.2 | random |
| medium | `line-n8-t90-o80-s30057` | 30057 | 7 | 0.250 | 3 | 0.9 | 0.8 | line |
| medium | `random-n8-t90-o50-s30058` | 30058 | 7 | 0.250 | 4 | 0.9 | 0.5 | random |
| medium | `clustered-n8-t90-o50-s30059` | 30059 | 7 | 0.250 | 4 | 0.9 | 0.5 | clustered |
| medium | `random-n8-t90-o50-s30061` | 30061 | 7 | 0.250 | 3 | 0.9 | 0.5 | random |
| medium | `clustered-n8-t80-o100-s30062` | 30062 | 7 | 0.250 | 3 | 0.8 | 1.0 | clustered |
| medium | `line-n8-t90-o80-s30063` | 30063 | 7 | 0.250 | 3 | 0.9 | 0.8 | line |
| high | `random-n8-t90-o80-s30015` | 30015 | 13 | 0.464 | 3 | 0.9 | 0.8 | random |
| high | `clustered-n8-t90-o80-s30017` | 30017 | 15 | 0.536 | 3 | 0.9 | 0.8 | clustered |
| high | `clustered-n8-t90-o80-s30018` | 30018 | 14 | 0.500 | 3 | 0.9 | 0.8 | clustered |
| high | `clustered-n8-t100-o50-s30020` | 30020 | 14 | 0.500 | 3 | 1.0 | 0.5 | clustered |
| high | `clustered-n8-t90-o80-s30021` | 30021 | 14 | 0.500 | 3 | 0.9 | 0.8 | clustered |
| high | `uniform-n8-t100-o50-s30024` | 30024 | 14 | 0.500 | 4 | 1.0 | 0.5 | uniform |
| high | `clustered-n8-t90-o50-s30026` | 30026 | 13 | 0.464 | 3 | 0.9 | 0.5 | clustered |
| high | `uniform-n8-t90-o50-s30027` | 30027 | 14 | 0.500 | 4 | 0.9 | 0.5 | uniform |
| high | `clustered-n8-t90-o80-s30028` | 30028 | 15 | 0.536 | 3 | 0.9 | 0.8 | clustered |
| high | `clustered-n8-t100-o50-s30029` | 30029 | 13 | 0.464 | 4 | 1.0 | 0.5 | clustered |
| high | `clustered-n8-t80-o100-s30030` | 30030 | 15 | 0.536 | 3 | 0.8 | 1.0 | clustered |
| high | `clustered-n8-t90-o80-s30031` | 30031 | 14 | 0.500 | 3 | 0.9 | 0.8 | clustered |
| high | `random-n8-t100-o20-s30032` | 30032 | 14 | 0.500 | 3 | 1.0 | 0.2 | random |
| high | `clustered-n8-t90-o80-s30033` | 30033 | 14 | 0.500 | 3 | 0.9 | 0.8 | clustered |
| high | `random-n8-t90-o50-s30034` | 30034 | 14 | 0.500 | 4 | 0.9 | 0.5 | random |
| high | `line-n8-t100-o80-s30035` | 30035 | 14 | 0.500 | 3 | 1.0 | 0.8 | line |
| high | `line-n8-t100-o50-s30037` | 30037 | 13 | 0.464 | 4 | 1.0 | 0.5 | line |
| high | `random-n8-t90-o50-s30038` | 30038 | 13 | 0.464 | 4 | 0.9 | 0.5 | random |
| high | `line-n8-t100-o50-s30046` | 30046 | 13 | 0.464 | 4 | 1.0 | 0.5 | line |
| high | `uniform-n8-t100-o50-s30050` | 30050 | 13 | 0.464 | 4 | 1.0 | 0.5 | uniform |

## Seeds

60 instances, 60 distinct seeds, range 30000–30063, reserved range 30000–39999.

