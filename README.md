# FC30 QRE v7 工作修订版

基于 `FC30_QRE_剩余窗口_v6` 修订。口径依据为用户 2026-10-10 逐项回复 Q1–Q16。
**v6 的三阶段剩余窗口聚合口径一字未改**，本轮改动集中在公式—代码一致性与
信息可追溯性。

## 运行

Python 3.10 以上：

```bash
python -m pip install -r requirements.txt

# 全量重算（默认）
python simulate_fc30.py --mode full --out results

# 仅基准 + 各对照轨
python simulate_fc30.py --mode baseline --out results_baseline

# 校验
python verify_model.py
```

### 单轨试算

```bash
python simulate_fc30.py --safety-form continuous --sigma-p 0.5 --mode baseline --out results_con
python simulate_fc30.py --safety-form paper_linear --mode baseline --out results_pl
```

`--safety-form` 取值 `saturating`（v6 基准）/ `continuous`（params 原字段）/
`paper_linear`（原文式迁移，扩展模型）。

### 事件分支

```bash
python simulate_fc30.py --events-json events_example.json --mode baseline --out results_event_demo
```

示例为非实测事件；有效事件应经外部监测或轨迹验证。接口只截断行为分支。
有效事件后的主 NN 物理字段为反事实审计，标准动作物理风险独立计算。

---

## 三条结果轨（Q1）

**主形式尚未指定。** 三种安全损失形式在同一场景、参数、轨迹、采样下并列输出：

| 形式 | 公式 | 来源 |
|---|---|---|
| `saturating` | `Cs = min(P/Pref, 1)` | 重修版第7节（v6 实现口径） |
| `continuous` | `Cs = P/(P+Pref)` | params `utility.formula` 原字段 |
| `paper_linear` | `Cs = (a·danger + b·tq)/(a+b)` | 原文式(10)(11) 迁移，**扩展模型** |

**为什么三轨并存**：v6 中 `params.utility.formula` 声明的是 `continuous`，
代码实现的是 `saturating`，二者不一致。本轮不静默覆盖任一方，而是并列计算。

选定主形式应依据迁移逻辑、变量解释与验证结果，**不以曲线是否上升为标准**。

---

## 四类口径分别报告（Q16）

| 口径 | 字段 |
|---|---|
| 在线行为指标 | `P_three_stage_dynamic` |
| 全历史回顾值 | `P_three_stage_full_history_retrospective` |
| 剩余阶段指数 | `R_remaining_stage_index` |
| 物理碰撞概率 | `A_Pcol_max_NN` 等 |

四者含义不同，不得互相替代。

**边界跳变诊断不再是正式风险结果。** v6 的"截止左极限"结论与命名已删除——
实测表明该跳变源于安全损失饱和截断穿越，不是阶段均值的数学左极限。

---

## 边界判据（Q3）

**两个参数已分开，不得因数值相同而混用：**

| 参数 | 角色 | 取值 |
|---|---|---|
| `Pref` | 归一化尺度 | 0.02 |
| `P_max` | 可接受碰撞概率阈值 | 0.02（**假设情景，无外部依据**） |

- 几何判据（**主**）：`min_q_safe > 1` 且恢复完成 → `t_f = 28.4 s`
- 概率判据（对照轨）：`Pcol_max < P_max` → `t_f = 28.3 s`

两者并列报告，不合并、不互相否定。

---

## 优先查看

| 文件 | 内容 |
|---|---|
| `模型说明与实验结果.md` | 逐项落实记录、公式量纲来源、实测数据、修正记录 |
| `simulate_fc30.py` | 主程序（安全损失形式可切换） |
| `verify_model.py` | 155 项校验，按四类实测分类 |
| `params_fc30_v3.json` | 参数；`utility.formula` 已补三轨说明 |
| `model_supplement.json` | 实现补充与 Q1–Q16 口径记录 |
| `results/safety_form_comparison.csv` | 三形式逐时刻并列 |
| `results/sigma_factorial_comparison.csv` | σ × 形式全因子 |
| `results/asymmetric_single_factor.csv` | 单因素非对称（B执行延迟） |
| `results/prediction_horizon_sensitivity.csv` | H 敏感性（t_c 固定） |
| `results/cutoff_boundary_numeric_diagnostic.csv` | 边界跳变诊断（7 个 ε 探针） |
| `results/lambda_sweep.csv` | 全部候选根 + 正反向扫描对照 |
| `results/stage_factor_provenance.csv` | 每时刻三因子的来源、时间窗、样本数 |

---

## 本轮修复的实现问题

1. **`perceived()` 中 `qB` 漏接**：`qB` 由 `pB` 与 `hB` 派生，位置偏置 `pb` 作用于
   `pB` 却只对 `player=='B'` 作用于 `qB`，处理不一致。已改为一律同步平移。
2. **`paper_linear` 索引错误**：原实现取 `gate()` 返回 4 元组却按 2 元组解包，
   且未使用 commanded trajectory。已修正为基于 `min_q_collision`。

---

## 校验口径（Q13）

155 项全部通过，按**实际执行内容**分类：

| 类别 | 数量 |
|---|---:|
| analytic | 126 |
| self_consistency | 22 |
| convergence | 4 |
| simulated_input | 3 |

**通过这些检查不等于模型已获外部验证。** 它只表明实现与其自身定义及所述解析
参照一致。`self_consistency` 类检查只能验证定义被正确执行，不能判断定义本身
是否合理。

---

## 已知限制

- 能耗项**未实现**（v3 曾称"保留接口"，与代码不符，已更正）
- `P_max` 阈值**无外部依据**，为明确标注的假设情景
- `paper_linear` 轨在正碰场景下 danger 项接近二值，时间梯度由 `tq` 承担
- 三阶段个体均值相对离散度 1.57%，已定位为安全损失无梯度所致，只作诊断
- 阶段采样已收敛（dt 0.5→0.25 时 stage3 均值仅变 0.005%），非采样不足