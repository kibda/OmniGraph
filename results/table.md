# Results

Test-set score, mean +/- std over 3 seeds. Each domain uses its own
headline metric, so numbers are comparable **down** a column and
never **across** domains.

Arm D is collapsed to its best single source per cell -- the fair
comparison for arm A is the strongest expert, not an average one.

## cora  (accuracy)

| arm | 1% | 5% | 10% | 50% | 100% |
|---|---|---|---|---|---|
| B random-init | 0.434 ± 0.086 | 0.585 ± 0.071 | 0.612 ± 0.011 | 0.732 ± 0.024 | 0.763 ± 0.007 |
| D single-source expert | 0.411 ± 0.051 | 0.594 ± 0.034 | 0.645 ± 0.024 | 0.763 ± 0.015 | 0.796 ± 0.016 |
| A transfer (3 sources) | 0.392 ± 0.033 | 0.552 ± 0.029 | 0.619 ± 0.008 | 0.775 ± 0.018 | 0.800 ± 0.012 |
| C from-scratch | 0.489 ± 0.050 | 0.692 ± 0.012 | 0.734 ± 0.019 | 0.830 ± 0.006 | 0.846 ± 0.014 |

## elliptic  (auc_pr)

| arm | 1% | 5% | 10% | 50% | 100% |
|---|---|---|---|---|---|
| B random-init | 0.342 ± 0.070 | 0.536 ± 0.011 | 0.593 ± 0.016 | 0.672 ± 0.020 | 0.675 ± 0.015 |
| D single-source expert | 0.421 ± 0.023 | 0.568 ± 0.042 | 0.615 ± 0.021 | 0.659 ± 0.051 | 0.669 ± 0.053 |
| A transfer (3 sources) | 0.382 ± 0.021 | 0.497 ± 0.047 | 0.569 ± 0.063 | 0.597 ± 0.059 | 0.644 ± 0.031 |
| C from-scratch | 0.623 ± 0.044 | 0.806 ± 0.008 | 0.846 ± 0.016 | 0.922 ± 0.007 | 0.949 ± 0.005 |

## photo  (accuracy)

| arm | 1% | 5% | 10% | 50% | 100% |
|---|---|---|---|---|---|
| B random-init | 0.703 ± 0.018 | 0.841 ± 0.010 | 0.848 ± 0.006 | 0.873 ± 0.014 | 0.880 ± 0.011 |
| D single-source expert | 0.708 ± 0.011 | 0.823 ± 0.020 | 0.844 ± 0.016 | 0.874 ± 0.011 | 0.881 ± 0.008 |
| A transfer (3 sources) | 0.651 ± 0.009 | 0.819 ± 0.015 | 0.845 ± 0.007 | 0.881 ± 0.008 | 0.886 ± 0.009 |
| C from-scratch | 0.685 ± 0.023 | 0.866 ± 0.012 | 0.894 ± 0.008 | 0.925 ± 0.005 | 0.938 ± 0.006 |

## ppi  (micro_f1)

| arm | 1% | 5% | 10% | 50% | 100% |
|---|---|---|---|---|---|
| B random-init | 0.494 ± 0.002 | 0.522 ± 0.001 | 0.529 ± 0.005 | 0.532 ± 0.001 | 0.532 ± 0.003 |
| D single-source expert | 0.495 ± 0.008 | 0.523 ± 0.002 | 0.529 ± 0.002 | 0.530 ± 0.003 | 0.531 ± 0.005 |
| A transfer (3 sources) | 0.491 ± 0.009 | 0.521 ± 0.004 | 0.524 ± 0.002 | 0.527 ± 0.003 | 0.528 ± 0.005 |
| C from-scratch | 0.550 ± 0.007 | 0.637 ± 0.001 | 0.695 ± 0.010* | 0.869 ± 0.004* | 0.912 ± 0.004* |

`*` marks a cell where at least one run did not reach a
validation plateau; its score is a lower bound.
