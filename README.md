# UHackthon2022

This repo is my baseline on 2022UHackthon Question 3 [here](https://docs.qq.com/doc/DUUNuWUJLd2FpZmN5). And the .py file can be executed in the Official platform. Anyway, I am not willing to share my training set (it seems work with generated dataset). XD

## EDA & issues
- few samples
  - web scrapping?
  - data augmentation?
  - linear models?
- left join `info` and `sales`
- dirty data
  - `uuid` is not unique in `info`.
  - 44 `material_name` are duplicated.
  - 12 sales records are not in `info` table (if we cannot find them from website, drop them), BUT whether this scenario is in test set. 

## BaselineV1

\#lgbm \#mse->2.04 
- seems only 162 available samples after pre-processing
- the available training set for validation
- `channel`: EC->0, DT->1
- `bar_code`: ignore // all of them are 690...
- `ingredient`: union set
- `sales_value`: prediction target
- `launch_date`: extract yy, mm, dd and so on
- `sales_period_`: 6 OR 12
