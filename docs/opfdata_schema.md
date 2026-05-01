# JSON Schema

## grid/nodes/bus (4 features)

| Index | Feature    | Description                              | Units           |
| ----- | ---------- | ---------------------------------------- | --------------- |
| 0     | `base_kv`  | Base voltage                             | kV              |
| 1     | `bus_type` | Bus type (1=PQ, 2=PV, 3=Ref, 4=Isolated) | -               |
| 2     | `vmin`     | Minimum voltage magnitude                | per unit (p.u.) |
| 3     | `vmax`     | Maximum voltage magnitude                | per unit (p.u.) |

## grid/nodes/generator (11 features)

| Index | Feature   | Description                   | Units    |
| ----- | --------- | ----------------------------- | -------- |
| 0     | `mbase`   | Machine base power            | MVA      |
| 1     | `pg`      | Active power generation       | p.u      |
| 2     | `pmin`    | Minimum active power output   | p.u      |
| 3     | `pmax`    | Maximum active power output   | p.u      |
| 4     | `qg`      | Reactive power generation     | p.u      |
| 5     | `qmin`    | Minimum reactive power output | p.u      |
| 6     | `qmax`    | Maximum reactive power output | p.u      |
| 7     | `vg`      | Voltage setpoint              | per unit |
| 8     | `cost_c2` | Quadratic cost coefficient    | $/MW²/h  |
| 9     | `cost_c1` | Linear cost coefficient       | $/MW/h   |
| 10    | `cost_c0` | Constant cost coefficient     | $/h      |

## grid/nodes/load (2 features)

| Index | Feature | Description           | Units |
| ----- | ------- | --------------------- | ----- |
| 0     | `pd`    | Active power demand   | p.u   |
| 1     | `qd`    | Reactive power demand | p.u   |

## grid/nodes/shunt (2 features)

| Index | Feature | Description       | Units    |
| ----- | ------- | ----------------- | -------- |
| 0     | `bs`    | Shunt susceptance | per unit |
| 1     | `gs`    | Shunt conductance | per unit |

## grid/edges/ac_line/features (9 features)

| Index | Feature  | Description                      | Units    |
| ----- | -------- | -------------------------------- | -------- |
| 0     | `angmin` | Minimum voltage angle difference | radians  |
| 1     | `angmax` | Maximum voltage angle difference | radians  |
| 2     | `b_fr`   | Charging susceptance (from side) | per unit |
| 3     | `b_to`   | Charging susceptance (to side)   | per unit |
| 4     | `br_r`   | Series resistance                | per unit |
| 5     | `br_x`   | Series reactance                 | per unit |
| 6     | `rate_a` | Long-term thermal rating         | MVA      |
| 7     | `rate_b` | Short-term thermal rating        | MVA      |
| 8     | `rate_c` | Emergency thermal rating         | MVA      |

## grid/edges/transformer/features (11 features)

| Index | Feature  | Description                      | Units    |
| ----- | -------- | -------------------------------- | -------- |
| 0     | `angmin` | Minimum voltage angle difference | radians  |
| 1     | `angmax` | Maximum voltage angle difference | radians  |
| 2     | `br_r`   | Series resistance                | per unit |
| 3     | `br_x`   | Series reactance                 | per unit |
| 4     | `rate_a` | Long-term thermal rating         | MVA      |
| 5     | `rate_b` | Short-term thermal rating        | MVA      |
| 6     | `rate_c` | Emergency thermal rating         | MVA      |
| 7     | `tap`    | Transformer tap ratio            | per unit |
| 8     | `shift`  | Phase shift angle                | radians  |
| 9     | `b_fr`   | Charging susceptance (from side) | per unit |
| 10    | `b_to`   | Charging susceptance (to side)   | per unit |

## solution/nodes/bus (2 features)

| Index | Feature | Description       | Units    |
| ----- | ------- | ----------------- | -------- |
| 0     | `va`    | Voltage angle     | radians  |
| 1     | `vm`    | Voltage magnitude | per unit |

## solution/nodes/generator (2 features)

| Index | Feature | Description               | Units |
| ----- | ------- | ------------------------- | ----- |
| 0     | `pg`    | Active power generation   | p.u   |
| 1     | `qg`    | Reactive power generation | p.u   |

## solution/edges/ac_line/features (4 features)

| Index | Feature | Description                     | Units |
| ----- | ------- | ------------------------------- | ----- |
| 0     | `pt`    | Active power flow (to side)     | MW    |
| 1     | `qt`    | Reactive power flow (to side)   | MVAr  |
| 2     | `pf`    | Active power flow (from side)   | MW    |
| 3     | `qf`    | Reactive power flow (from side) | MVAr  |

## solution/edges/transformer/features (4 features)

Same as ac_line.

## HDF5 Schema

- Single scenario:

```bash
scenario_XXXXXX.h5
├── grid/                    (Group: Input network data)
├── solution/                (Group: Optimization results)
└── metadata/                (Group: Scenario metadata)
```

- Multiple scenarios:

```bash
chunk_XXXX.h5
├── scenario_000001/
│   ├── grid/
│   ├── solution/
│   └── metadata/
├── scenario_000002/
│   ├── grid/
│   ├── solution/
│   └── metadata/
...
└── scenario_00YYYY/
├── grid/
├── solution/
└── metadata/
```

### Field specifications:

#### grid/nodes/bus (Dataset: Float32, shape=[n_bus, 5])

| Index | Feature    | Type    | Description                  | Units           |
| ----- | ---------- | ------- | ---------------------------- | --------------- |
| 0     | `vmin`     | Float32 | Minimum voltage magnitude    | per unit (p.u.) |
| 1     | `vmax`     | Float32 | Maximum voltage magnitude    | per unit (p.u.) |
| 2     | `zone`     | Float32 | Zone identifier              | -               |
| 3     | `area`     | Float32 | Area identifier              | -               |
| 4     | `bus_type` | Float32 | Bus type (1=PQ, 2=PV, 3=Ref) | -               |


#### grid/nodes/generator (Dataset: Float32, shape=[n_gen, 10])

| Index | Feature      | Type    | Description                         | Units    |
| ----- | ------------ | ------- | ----------------------------------- | -------- |
| 0     | `pmax`       | Float32 | Maximum active power output         | MW       |
| 1     | `pmin`       | Float32 | Minimum active power output         | MW       |
| 2     | `qmax`       | Float32 | Maximum reactive power output       | MVAr     |
| 3     | `qmin`       | Float32 | Minimum reactive power output       | MVAr     |
| 4     | `cost_c2`    | Float32 | Quadratic cost coefficient (c2·pg²) | $/MW²/h  |
| 5     | `cost_c1`    | Float32 | Linear cost coefficient (c1·pg)     | $/MW/h   |
| 6     | `cost_c0`    | Float32 | Constant cost coefficient           | $/h      |
| 7     | `vg`         | Float32 | Voltage setpoint (for PV buses)     | per unit |
| 8     | `mbase`      | Float32 | Machine base power                  | MVA      |
| 9     | `gen_status` | Float32 | Generator status (1=on, 0=off)      | -        |


#### grid/nodes/load (Dataset: Float32, shape=[n_load, 2])

| Index | Feature | Type    | Description           | Units |
| ----- | ------- | ------- | --------------------- | ----- |
| 0     | `pd`    | Float32 | Active power demand   | MW    |
| 1     | `qd`    | Float32 | Reactive power demand | MVAr  |


#### grid/nodes/shunt (Dataset: Float32, shape=[n_shunt, 2])

| Index | Feature | Type    | Description       | Units    |
| ----- | ------- | ------- | ----------------- | -------- |
| 0     | `gs`    | Float32 | Shunt conductance | per unit |
| 1     | `bs`    | Float32 | Shunt susceptance | per unit |

#### grid/context/baseMVA` (Dataset: Float32, shape=[1, 1, 1])

#### grid/edges/ac_line/
- `senders` (Dataset: Int32, shape=[n_ac_line])
- `receivers` (Dataset: Int32, shape=[n_ac_line])
- `features` (Dataset: Float32, shape=[n_ac_line, 9])

| Index | Feature     | Type    | Description                         | Units    |
| ----- | ----------- | ------- | ----------------------------------- | -------- |
| 0     | `angmin`    | Float32 | Minimum voltage angle difference    | radians  |
| 1     | `angmax`    | Float32 | Maximum voltage angle difference    | radians  |
| 2     | `br_r`      | Float32 | Series resistance                   | per unit |
| 3     | `br_x`      | Float32 | Series reactance                    | per unit |
| 4     | `b_fr`      | Float32 | Half of br_b                        | per unit |
| 5     | `b_to`      | Float32 | Half of br_b                        | per unit |
| 6     | `rate_a`    | Float32 | Long-term thermal rating            | MVA      |
| 7     | `rate_b`    | Float32 | Short-term thermal rating           | MVA      |
| 8     | `rate_c`    | Float32 | Emergency thermal rating            | MVA      |
| 9     | `br_status` | Float32 | Branch status (1=in-service, 0=out) | -        |


#### grid/edges/transformer/

- `senders` (Dataset: Int32, shape=[n_transformer])
- `receivers` (Dataset: Int32, shape=[n_transformer])
- `features` (Dataset: Float32, shape=[n_transformer, 11])

| Index | Feature     | Type    | Description                      | Units    |
| ----- | ----------- | ------- | -------------------------------- | -------- |
| 0     | `angmin`    | Float32 | Minimum voltage angle difference | radians  |
| 1     | `angmax`    | Float32 | Maximum voltage angle difference | radians  |
| 2     | `br_r`      | Float32 | Series resistance                | per unit |
| 3     | `br_x`      | Float32 | Series reactance                 | per unit |
| 4     | `b_fr`      | Float32 | Half of br_b                     | per unit |
| 5     | `b_to`      | Float32 | Half of br_b                     | per unit |
| 6     | `rate_a`    | Float32 | Long-term thermal rating         | MVA      |
| 7     | `rate_b`    | Float32 | Short-term thermal rating        | MVA      |
| 8     | `rate_c`    | Float32 | Emergency thermal rating         | MVA      |
| 9     | `br_status` | Float32 | Branch status                    | -        |
| 10    | `tap`       | Float32 | Transformer tap ratio            | per unit |
| 11    | `shift`     | Float32 | Phase shift angle                | radians  |

- `solution/nodes/bus` (Dataset: Float32, shape=[n_bus, 2])

| Index | Feature | Type    | Description       | Units    |
| ----- | ------- | ------- | ----------------- | -------- |
| 0     | `va`    | Float32 | Voltage angle     | radians  |
| 1     | `vm`    | Float32 | Voltage magnitude | per unit |

- `solution/nodes/generator` (Dataset: Float32, shape=[n_gen, 2])

| Index | Feature | Type    | Description               | Units |
| ----- | ------- | ------- | ------------------------- | ----- |
| 0     | `pg`    | Float32 | Active power generation   | MW    |
| 1     | `qg`    | Float32 | Reactive power generation | MVAr  |


- `solution/edges/ac_line/features` (Dataset: Float32, shape=[n_ac_line, 4])
Power flows on AC lines.

| Index | Feature | Type    | Description                     | Units |
| ----- | ------- | ------- | ------------------------------- | ----- |
| 0     | `pf`    | Float32 | Active power flow (from → to)   | MW    |
| 1     | `qf`    | Float32 | Reactive power flow (from → to) | MVAr  |
| 2     | `pt`    | Float32 | Active power flow (to → from)   | MW    |
| 3     | `qt`    | Float32 | Reactive power flow (to → from) | MVAr  |

- `solution/edges/ac_line/features` (Dataset: Float32, shape=[n_ac_line, 4])

| Index | Feature | Type    | Description                     | Units |
| ----- | ------- | ------- | ------------------------------- | ----- |
| 0     | `pf`    | Float32 | Active power flow (from → to)   | MW    |
| 1     | `qf`    | Float32 | Reactive power flow (from → to) | MVAr  |
| 2     | `pt`    | Float32 | Active power flow (to → from)   | MW    |
| 3     | `qt`    | Float32 | Reactive power flow (to → from) | MVAr  |

- `solution/edges/transformer/features` (Dataset: Float32, shape=[n_transformer, 4])

| Index | Feature | Type    | Description                     | Units |
| ----- | ------- | ------- | ------------------------------- | ----- |
| 0     | `pf`    | Float32 | Active power flow (from → to)   | MW    |
| 1     | `qf`    | Float32 | Reactive power flow (from → to) | MVAr  |
| 2     | `pt`    | Float32 | Active power flow (to → from)   | MW    |
| 3     | `qt`    | Float32 | Reactive power flow (to → from) | MVAr  |

- `metadata/` Attributes

| Attribute           | Type    | Description                              | Units      |
| ------------------- | ------- | ---------------------------------------- | ---------- |
| `scenario_id`       | Int32   | Unique scenario identifier               | -          |
| `objective`         | Float32 | Optimal objective function value         | $ (or $/h) |
| `solve_time`        | Float32 | Solver wall-clock time                   | seconds    |
| `status`            | String  | Solver termination status                | -          |
| `total_power_slack` | Float32 | Sum of all power balance slack variables |
| `total_line_slack`  | Float32 | Sum of all line limit slack variables    |


## Contingency HDF5 schemas:

### Single scenario:

```bash
scenario_XXXXXX.h5
├── grid/                    (Group: Input network with perturbed loads)
├── base_solution/           (Group: Pre-contingency OPF solution)
├── contingencies/           (Group: Contingency definitions)
├── post_contingency/        (Group: Post-contingency OPF solutions)
└── metadata/                (Group: Scenario metadata)
```

### Multiple scenarios:

```bash
merged_XXXXXXX.h5
├── grid/
├── scenario_000001/
│   ├── base_solution/
│   ├── contingencies/
│   ├── post_contingency/
└   ├── metadata/  
├── scenario_000002/
│   ...
```

- `grid/nodes/load` (Dataset: Float32, shape=[n_load, 4])

| Index | Feature    | Description                       | Units |
| ----- | ---------- | --------------------------------- | ----- |
| 0     | `pd`       | Active power demand               | MW    |
| 1     | `qd`       | Reactive power demand             | MVAr  |
| 2     | `weight_p` | Load shedding priority weight (P) | -     |
| 3     | `weight_q` | Load shedding priority weight (Q) | -     |

```bash
base_solution/
├── pf/nodes/          (Power Flow solution)
│   ├── bus            [n_bus, 2]
│   ├── generator      [n_gen, 2]
│   └── load           [n_load, 2]
└── opf/nodes/         (Optimal Power Flow solution)
├── bus            [n_bus, 2]
├── generator      [n_gen, 2]
└── load           [n_load, 2]
```

##### `bus` (Dataset: Float32, shape=[n_bus, 2])
| Index | Feature | Description       | Units    |
| ----- | ------- | ----------------- | -------- |
| 0     | `va`    | Voltage angle     | radians  |
| 1     | `vm`    | Voltage magnitude | per unit |

##### `generator` (Dataset: Float32, shape=[n_gen, 2])
| Index | Feature | Description               | Units |
| ----- | ------- | ------------------------- | ----- |
| 0     | `pg`    | Active power generation   | MW    |
| 1     | `qg`    | Reactive power generation | MVAr  |

##### `load` (Dataset: Float32, shape=[n_load, 2])
| Index | Feature     | Description                    | Units |
| ----- | ----------- | ------------------------------ | ----- |
| 0     | `pd_served` | Active power actually served   | MW    |
| 1     | `qd_served` | Reactive power actually served | MVAr  |

#### Attributes (`base_solution/opf/`)
| Attribute    | Type    | Description                                      |
| ------------ | ------- | ------------------------------------------------ |
| `objective`  | Float32 | Optimal cost                                     |
| `solve_time` | Float32 | Solver time (seconds)                            |
| `status`     | String  | Solver termination status (e.g., LOCALLY_SOLVED) |

### CONTINGENCIES GROUP
Definitions of all contingencies analyzed.

#### Attributes
| Attribute | Type  | Description             |
| --------- | ----- | ----------------------- |
| `count`   | Int32 | Number of contingencies |

#### Datasets

##### `contingencies/types` (Dataset: Int8, shape=[n_cont])
Contingency type for each contingency.

| Value | Type | Description        |
| ----- | ---- | ------------------ |
| 0     | LINE | Line/branch outage |
| 1     | GEN  | Generator outage   |

- `contingencies/ids` (Dataset: String, shape=[n_cont])
- `contingencies/names` (Dataset: String, shape=[n_cont])

```bash
post_contingency/
├── contingency_000001/
│   ├── @pf_converged      Int8 (1=yes, 0=no)
│   ├── @opf_converged     Int8 (1=yes, 0=no)
│   ├── pf/nodes/
│   │   ├── bus            [n_bus, 2]
│   │   ├── generator      [n_gen, 2]
│   │   └── load           [n_load, 2]
│   └── opf/
│       ├── @objective     Float32
│       ├── @solve_time    Float32
│       ├── @status        String
│       └── nodes/
│           ├── bus        [n_bus, 2]
│           ├── generator  [n_gen, 2]
│           └── load       [n_load, 2]
├── contingency_000002/
│   └── ...
└── contingency_NNNNNN/
```

#### Contingency Attributes (`post_contingency/contingency_XXXXXX/`)
| Attribute       | Type | Description                     |
| --------------- | ---- | ------------------------------- |
| `pf_converged`  | Int8 | 1 if PF converged, 0 otherwise  |
| `opf_converged` | Int8 | 1 if OPF converged, 0 otherwise |

#### OPF Attributes (`post_contingency/contingency_XXXXXX/opf/`)
| Attribute    | Type    | Description                                      |
| ------------ | ------- | ------------------------------------------------ |
| `objective`  | Float32 | Optimal cost                                     |
| `solve_time` | Float32 | Solver time (seconds)                            |
| `status`     | String  | Solver termination status (e.g., LOCALLY_SOLVED) |

#### Node Datasets (same structure as base_solution)
- `bus` [n_bus, 2]: va, vm
- `generator` [n_gen, 2]: pg, qg
- `load` [n_load, 2]: pd_served, qd_served

---

### METADATA GROUP (Attributes)

| Attribute          | Type    | Description                      |
| ------------------ | ------- | -------------------------------- |
| `scenario_id`      | Int32   | Unique scenario identifier       |
| `n_contingencies`  | Int32   | Total contingencies analyzed     |
| `n_pf_converged`   | Int32   | Number of PF that converged      |
| `n_opf_converged`  | Int32   | Number of OPF that converged     |
| `total_solve_time` | Float32 | Sum of all solve times (seconds) |
