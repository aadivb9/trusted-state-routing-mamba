The post-development closing diagnostic completed all 600 reserved baseline examples. It retains the original failed competence gate and does not include attack-conditioned generations.

| Role | Task | Published parser correct | Strict parser correct | Strict valid |
| --- | --- | ---: | ---: | ---: |
| Target | sst2 | 39/100 | 12/100 | 16/100 |
| Target | sms_spam | 23/100 | 0/100 | 1/100 |
| Target | mrpc | 50/100 | 0/100 | 0/100 |
| Injected | sst2 | 53/100 | 58/100 | 71/100 |
| Injected | sms_spam | 11/100 | 0/100 | 0/100 |
| Injected | mrpc | 49/100 | 0/100 | 0/100 |

The target role measures the legitimate task on its reserved inputs. The injected role measures the attacker task on its own clean inputs, not attack success. The roles differ in source records and published instruction variants, so their difference does not isolate a causal prompt effect. Each task/role sample is balanced and selected by the published deterministic procedure. These are descriptive results for the selected records, not random population samples. Formatting and task dependence must be considered alongside accuracy.
