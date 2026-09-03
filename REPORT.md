# Agnes 2B: A Reproducible and Cost-Aware Two-Stage Pretraining Protocol on Consumer GPU Clusters

> Faithful, page-aligned text transcription of the supplied technical paper. The PDF is the canonical visual publication; this Markdown version is provided for direct reading and search on GitHub.

## Full Text

### Page 1

```text
TECHNICAL REPORT V1.1
Agnes 2B: A Reproducible and Cost-Aware
Two-Stage Pretraining Protocol on Consumer
GPU Clusters
Agnes Foundation Model Team
Agnes AI · September 2026
Abstract
We present Agnes 2B, a reproducible pretraining protocol for a compact dense language
model on consumer-grade GPU clusters. The proposed model contains approximately 2.032
billion parameters, uses 28 decoder blocks and a 4,096-token training context, and is trained
on 1.3988 trillion tokens in two stages. The data schedule moves from broad language cov-
erage toward greater emphasis on mathematics, code, and instruction-formatted data while
retaining controlled replay. The training method combines blockwise E4M3 FP8 linear opera-
tions, BF16/FP32 computation for numerically sensitive components, a hybrid MuonH–AdamW
optimizer, two-stage learning-rate scheduling, source-local curriculum ordering, and late-stage
checkpoint weight averaging. The distributed configuration scales from 24 to 96 RTX 5090
GPUs using tensor parallelism of one and topology-aware pipeline and data parallelism. Based
on measurements from a structurally matched reference system, the full-scale run is projected to
require 22,514 active GPU-hours and 17.59 days of pure training time. The capability plan uses
reference-reported endpoints of 43.50, 63.02, and 57.81 for Math/Code A vg4, Reason/Knowledge
A vg11, and A vg15, while Agnes preregisters release floors of 42.0, 62.5, and 57.0, together
with final validation loss at most 2.52. We further specify controlled proxy studies, ablations,
evaluation metrics, recovery requirements, and preregistered acceptance criteria. This paper
defines a falsifiable training protocol rather than reporting completed Agnes training results.
Keywords: foundation-model pretraining; consumer GPU clusters; mixed-precision training; cur-
riculum learning; distributed training; reproducibility.
Evidence convention. Reported evidence identifies measurements from prior work; Agnes design
identifies a proposed Agnes configuration; and Release gate identifies a threshold that must be
measured before scale-up or release. Unless stated otherwise, cost denotes accelerator-equivalent
cost rather than total project expenditure.
T able 1. Core specification of the proposed Agnes 2B program.
Item Specified value Item Specified value
Model scale ∼2.032B, dense Training tokens 1.3988T
Training context 4,096 Global batch size 1,536 sequences
Phase 1 24 GPUs, 438.84B Phase 2 96 GPUs, 959.99B
Primary arithmetic Blockwise E4M3 FP8 Persistent weights BF16
Production GPU-hours 22,514, planning anchor Pure training time 17.59 days, planning
anchor
Primary evaluation 15-task suite plus
Chinese evaluation
Final performance
floor
Loss ≤ 2.52; A vg15 ≥
57.0; A vg4 ≥ 42.0; A vg11
≥ 62.5
```

### Page 2

```text
Agnes 2B Pretraining Protocol Technical Report
1 Introduction
1.1 Motivation and Objectives
Agnes 2B is intended to be a locally deployable, continually trainable, and auditable general-purpose
base model. The pretraining program pursues four concurrent objectives: stable English and Chinese
knowledge coverage; a transferable foundation for mathematical reasoning and code generation;
reproducible throughput on consumer GPU clusters; and complete freezing of data manifests, sample
order, training state, and evaluation protocols so that subsequent supervised fine-tuning, preference
optimization, and agent training can be attributed to a well-defined base checkpoint.
Success is not defined merely as processing 1.4T tokens. A successful run must satisfy all of the
following within the approved budget: no unrecoverable optimization divergence; traceable data
lineage and licensing; validation loss within a prespecified scaling envelope; core capability above
release thresholds; and a final checkpoint whose tensor shapes, parameter grouping, hashes, and
recovery behavior can be reproduced from the released configuration.
1.2 Scope
The report covers tokenizer construction, data acquisition and governance, model architecture,
low-precision numerical execution, optimization, curriculum ordering, parallelism, failure recovery,
base-model evaluation, and the release package. It does not cover supervised fine-tuning, prefer-
ence optimization, reinforcement learning, tool-use policy, or product-level safety alignment; those
activities begin from the Base checkpoint produced here and require separate experimental plans.
The numerical budget applies only to the fixed dense architecture described in Section 3. It must
not be extrapolated directly to 7B, 120B, or mixture-of-experts models. The primary hardware
target is an eight-GPU node with 32GB RTX 5090 devices. Alternative accelerators require new
throughput, memory, communication, and wall-clock measurements.
1.3 Contributions and Non-Claims
This report makes four practical contributions. First, it converts favorable results from a low-cost
pretraining study [ 1] into explicit, testable design hypotheses. Second, it integrates model, data,
optimizer, arithmetic, and systems decisions into a single token-accounted training plan. Third, it
defines gates that can stop the project before the high-cost production run. Fourth, it specifies
reproducibility artifacts and evaluation rules before results are observed.
No performance number in this report is an Agnes result unless it is explicitly labeled as such in
a future revision. Values inherited from the reference study remain reported evidence; values such
as architecture settings, budgets, and thresholds are proposed decisions or release criteria. This
distinction prevents planning targets from being presented as empirical achievements.
2 Design Rationale and Evidence Mapping
2.1 Evidence-to-Decision Mapping
Table 2 summarizes how each major observation reported in the reference study [ 1] is converted into
an Agnes design choice and an independent pre-run test.
```

### Page 3

```text
Agnes 2B Pretraining Protocol Technical Report
T able 2. Reported evidence, Agnes adoption, and required revalidation.
Reported evidence Agnes design choice Validation before full training
FP8–BF16
validation-loss
difference of
0.0031–0.0039 and
1.36× throughput at
the 1.7B proxy scale
Use blockwise E4M3 from random
initialization while retaining sensitive
operators in BF16/FP32
Paired 170M and 600M runs with
identical data and seeds; loss
difference ≤ 0.005 and throughput
≥ 1.25×
Approximately 1.19 ×
quality-adjusted
compute gain from
the complete MuonH
recipe, based on short
scaling runs
Apply MuonH only to approximately
scale-invariant attention and
MLP matrices; optimize all other
parameters with AdamW
At least two model scales and two
seeds; compare against tuned AdamW
and standard-Muon baselines
Approximately
1.18 points of
average-score
improvement
from source-local
curriculum ordering
at a single endpoint
In Phase 2, order examples from
lower to higher preference within
each source; never compare raw scores
across sources
Train a uniform-order shadow branch
on the identical token multiset
and interpret any gain only after
contamination analysis
Production
configurations of
24 and 96 GPUs
sustained 238 and
192 TFLOP/s/GPU,
respectively
Use TP=1 with communication-aware
pipeline and data parallelism,
including asymmetric layer placement
Run a 1,000-step systems trial on
supported drivers; recompute time
and cost if throughput gates are
missed
USD 6.89K covers
active GPU
time after data
materialization only
Treat this figure as a
production-accelerator planning
anchor, never as the total project
price
Account separately for data,
experiments, storage, evaluation,
labor, compliance, and rerun capacity
Boundary of the evidence
The reference study [ 1] does not disclose the production tokenizer, every random seed,
the complete AdamW configuration, or several numerical-stability settings. Most eﬀiciency
conclusions were obtained from scaling runs near 20 tokens per parameter, whereas the
proposed production run is close to 700 tokens per parameter. Agnes therefore treats the
reported methods as high-value starting hypotheses, not as results that transfer without
verification.
2.2 Two-Stage Program Logic
The training program separates broad distributional coverage from targeted capability development.
Phase 1 establishes a robust language-model prior over English, Chinese, mathematics, and code.
Phase 2 increases the representation of mathematics, code, and instruction-like material; introduces
a smooth replay transition; imposes source-local curriculum order; and concludes with a constant-
learning-rate branch followed by checkpoint averaging. This staging allows the team to isolate
```

### Page 4

```text
Agnes 2B Pretraining Protocol Technical Report
distribution shifts, audit replay accounting, and retain fallback endpoints if the curriculum or late-
stage averaging does not generalize.
3 Model and Tokenizer Design
3.1 Architecture
Token IDs, positions, and document-level attention masks
Untied input embedding: V × d
28 × Agnes Decoder Block
RMSNorm → GQA → residual → RMSNorm → SwiGLU
Final RMSNorm
Untied LM head: d × V and next-token loss
Numerical path
Linear: FP8
Attention: BF16
Accumulation:
FP32
Training control
MuonH + AdamW
Power + linear LR
Figure 1 . Information flow and precision boundaries in Agnes 2B. The input embedding and output head
do not share weights.
T able 3. Proposed Agnes 2B architecture.
Configuration item Value Rationale or constraint
Model family Dense decoder-only Causal language modeling only
Layers / hidden size 28 / 2,048 Compatible with a mature 1.7B-class backbone
configuration [ 2]
FFN intermediate size 6,144 SwiGLU with three bias-free projections
Query / KV heads 16 / 8 Grouped-query attention; head dimension 128
Vocabulary size 151,936 Includes 256 reserved control tokens
Embedding / LM head Untied Approximately 311.16M parameters on each
side; pipeline placement must account for both
Normalization Pre-RMSNorm ϵ = 10 −6; optional head-wise RMSNorm on Q/K
Positional encoding RoPE θ = 10 6; train at 4,096 and preserve structural
capacity to 40,960
Activation / dropout SiLU / 0.0 Overfitting is addressed through data scale and
deduplication
Total parameters Approximately 2.032B Final tensor-by-tensor enumeration is
authoritative
Ignoring small normalization terms, the parameter count is approximated by
N ≈ 2V d + L

2d2 + 2d(Hkvdh) + 3ddf f

. (1)
With V = 151 ,936, d = 2 ,048, L = 28 , Hkv = 8 , dh = 128 , and df f = 6 ,144, Eq. ( 1) yields
approximately 2.03B parameters. The configuration gate nevertheless requires exact agreement
```

### Page 5

```text
Agnes 2B Pretraining Protocol Technical Report
among the exported parameter inventory, optimizer-group membership, and checkpoint tensor
shapes.
3.2 Training Objective, Initialization, and Packing
The objective is standard next-token cross-entropy,
L(θ) = − 1P
i mi
X
i
mi log pθ(xi | x<i), (2)
where mi masks padding, invalid positions after document boundaries, and control tokens excluded
from the loss. Length-aware packing is used for throughput. Documents within one packed sequence
receive a block-diagonal causal mask and independently reset position indices, preventing artificial
cross-document adjacency from becoming a learned regularity.
Linear layers use a truncated-normal initialization with baseline standard deviation 0.02.
Residual-branch output projections are additionally scaled by 1/
√
2L. Before scale-up, the team
must record layerwise activation standard deviation, maximum absolute value, and FP8 saturation
immediately after initialization. Any non-finite value blocks progression.
3.3 Tokenizer Construction
Because the reference study [ 1] does not specify its production tokenizer, Agnes treats tokenizer
development as a separate workflow that must finish before token budgets are frozen:
1. Sample approximately 100GB of text at document level from the final licensing whitelist,
following the target proportions for English, Chinese, mathematics, and code.
2. Train a byte-level BPE tokenizer with a total vocabulary of 151,936 tokens and byte fallback,
guaranteeing lossless round-trip behavior for arbitrary UTF-8 input.
3. Apply only UTF-8 validation and NFC normalization to natural language. Preserve indentation,
whitespace, and case in code; normalize line endings only.
4. Reserve 256 fixed token IDs inside the total vocabulary for BOS/EOS/PAD, fill-in-the-middle,
conversation boundaries, tools, and future multimodal placeholders. Inactive tokens are not
inserted into pretraining text.
5. Measure fertility, unknown-byte rate, and round-trip fidelity separately for all four domains.
Relative to the frozen comparison tokenizer, English, Chinese, and code fertility may not worsen
by more than 3%, and round-trip fidelity must be 100%.
Tokenizer freeze gate T0
Training-token ledgers may be recomputed only after the tokenizer model, normalizer, special-
token map, training-sample manifest, random seed, and SHA-256 hashes are frozen. Any
subsequent tokenizer change invalidates prior token counts and step estimates and therefore
requires rematerializing all shards.
```

### Page 6

```text
Agnes 2B Pretraining Protocol Technical Report
4 Data Mixture, Governance, and Curriculum
4.1 Domain Allocation Across Two Phases
Phase 1 prioritizes broad language-model coverage. Phase 2 increases the relative weight of mathe-
matics, code, and instruction-like material while using replay to avoid an abrupt distribution shift.
Table 4 is the initial materialization target. Proxy results may motivate changes within a domain,
but the phase-level token ledger must remain fixed after data gate D0.
T able 4. Domain-level token budgets for the two training phases.
Domain Phase 1 tokens Phase 1 share Phase 2 pool Phase 2 share
General English and knowledge 321.31B 73.2% 557.52B 59.4%
Mathematics and formal reasoning 31.56B 7.2% 171.29B 18.3%
General Chinese and knowledge 51.34B 11.7% 88.48B 9.4%
Code and software engineering 34.62B 7.9% 108.11B 11.5%
Instruction-like data 0 0.0% 12.66B 1.3%
Component-pool total 438.84B 100.0% 938.06B 100.0%
Displayed components are rounded to 0.01B tokens and shares to 0.1 percentage points; the phase totals are
authoritative.
Phase 1
Phase 2
0% 20% 40% 60% 80% 100%
English Math Chinese Code Instruction-like
Figure 2 . Domain mixture by phase. Phase 2 assigns substantially more capacity to mathematics and
code. Bars are normalized to 100% after rounding.
4.2 Sources and Governance Pipeline
Candidate sources should prioritize traceable open-web educational material, encyclopedic and
academic text, clearly licensed code, mathematical reasoning and proof data, Chinese knowledge
corpora, and limited quantities of tool-use, long-context, and terminal trajectories. A dataset name
is not itself a licensing conclusion. Every version must record its publisher, original URL, ver-
sion, license, redistribution restrictions, jurisdictional constraints, access date, and deletion-request
mechanism.
License
and source
whitelist
Parsing,
language ID,
PII and
safety filters
Source-
local dedup.
benchmark de-
contamination
Proxy
evaluation
source-local
selection
Static
materialization
manifest
+ hashes
Figure 3 . Agnes data-governance pipeline. Production training uses neither online scoring nor
dynamically changing mixtures.
The immutable processing order is license admission → document parsing → language and
domain classification → PII, malicious-code, and low-quality filtering → exact and MinHash dedu-
plication → benchmark decontamination → tokenization and packing → static shards. Ordinary
web corpora are first deduplicated within source. Benchmark questions, mirrored code repositories,
and heavily reused synthetic data additionally undergo cross-source near-duplicate detection. Every
shard records source-document IDs, the transformation chain, token counts, random seeds, and
hashes.
```

### Page 7

```text
Agnes 2B Pretraining Protocol Technical Report
4.3 Proxy-Based Data Selection
Agnes uses a shared 0.6B proxy checkpoint to measure the capability direction induced by each
candidate source; raw quality scores are never compared across unrelated sources. The proposed
protocol first trains a common starting point on 86B Phase 1 candidate tokens. Each candidate
condition then continues for approximately 8.4B tokens: 2,000 steps at sequence length 4,096 and
global batch size 1,024. The candidate-source share rises linearly from 0 to 80% over the first 1,600
steps and remains at 80% for the final 400 steps.
For scored sources larger than 50B tokens, the experiment evaluates the source-local q00, q25,
q50, and q75 slices. Sources containing 5B–50B tokens, or sources without a reliable score, con-
tribute a fixed sample of approximately 4B tokens. Each condition is reported as a four-axis
capability vector covering mathematics, code, Chinese, and general knowledge, with all task-level
scores retained. Principal component analysis may summarize capability direction but is not treated
as causal evidence. Selections among the five largest Phase 2 sources require two sample-order seeds.
When observed differences are smaller than evaluation noise, selection defaults to licensing clarity,
unique-token contribution, and processing cost.
4.4 Phase 2 Transition and Source-Local Curriculum
The Phase 2 component pool contains 938.06B tokens. An additional 21.93B tokens are replayed
from Phase 1, producing 959.99B physical Phase 2 tokens. The first 43.87B tokens form a linear
transition: the Phase 1 replay weight decreases from 100% to zero while the Phase 2 weight increases
from zero to 100%. Consequently, approximately 21.93B tokens come from replay and approximately
21.93B come from the beginning of the Phase 2 pool. The latter quantity must not be counted twice.
Following the curriculum-learning motivation in prior work [ 6], curriculum order is defined
only within each source. Components with a validated score are arranged from lower to higher
preference; components without a score use a fixed random permutation. Cumulative token mass
is mapped to 376 progress buckets, each containing approximately 2.5B tokens. Each bucket draws
the corresponding percentile interval from all components, approximately preserving the mixture
across sources. A uniform control branch uses the identical token multiset but applies a global
random shuffle.
Data freeze gate D0
Phase 1, the Phase 2 pool, and replay each require three independent accounting views: a
unique-document ledger, a materialized-token ledger, and a physical-training-token counter.
They must never be conflated. The published budget is defined by the materialized-token
ledger, while the running non-padding token counter provides real-time reconciliation. A
discrepancy above 0.1% automatically pauses training.
5 Optimization and Numerical Precision
5.1 Grouped Optimizers
Following the constrained matrix-optimization formulation of Wen et al. [ 5], MuonH updates only
explicitly whitelisted matrices in the attention and MLP modules. Embeddings, RMSNorm param-
eters, the LM head, biases, and all remaining tensors are optimized with AdamW. The MuonH
group uses zero weight decay and ten times the base learning-rate schedule. The AdamW group
uses weight decay 0.1, with proposed initialization values β1 = 0.9, β2 = 0.95, and ϵ = 10 −8. These
AdamW settings are Agnes engineering priors and require validation at 170M and 600M scales.
```

### Page 8

```text
Agnes 2B Pretraining Protocol Technical Report
For a selected matrix Wt, let R = ∥W0∥F and normalize the Muon update ut as but = ut/∥ut∥F .
The constrained update is
fWt+1 = Wt − ηtRbut, W t+1 = R
fWt+1
∥fWt+1∥F
. (3)
This update preserves matrix radius and makes ηt directly interpretable relative to weight scale. The
implementation must log weight norm, update norm, and effective learning rate for every parameter
group. A matrix that repeatedly triggers extreme projection or anomalous norms is removed from
the MuonH whitelist and retested from the most recent healthy checkpoint at proxy scale.
5.2 FP8 Mixed-Precision Path
T able 5. Proposed low-precision numerical path.
Operator or state Precision Rule
Q/K/V, attention-output,
and MLP projections
E4M3 FP8 GEMM Forward, data-gradient, and weight-gradient
GEMMs use blockwise execution
Activation / activation
gradient
FP8, 1 × 128 group Online scaling along the GEMM reduction
dimension
Weight quantization FP8, 128 × 128 block Scales restricted to powers of two; saturation is
logged
Core attention, residual
path, RMSNorm
BF16 Sensitive reductions remain outside FP8
Softmax, gradient
reduction, optimizer
accumulation
FP32 Master weights and optimizer states retain high
precision
Checkpoint weights BF16 Deployment quantization is a separate
post-training stage
The implementation follows blockwise FP8 execution guidance [ 4]. FP8 is enabled from random
initialization rather than after a fixed BF16 warmup, while per-layer BF16 fallback remains available.
Every 50 steps, the system records non-finite values, amax statistics, scales, saturation, gradient
norms, and loss spikes. Training pauses automatically if any critical layer exceeds 0.5% FP8
saturation for three consecutive observations, if paired validation loss differs by more than 0.005, or
if a non-finite gradient appears. Recovery begins from the preceding healthy checkpoint with the
implicated layers moved to BF16.
5.3 Two-Stage Learning-Rate Schedule
The maximum number of physical tokens per optimization step is
Btok = 1,536 × 4,096 = 6 ,291,456. (4)
Phase 1 therefore requires approximately 69,752 steps. The base learning rate warms up linearly for
1,000 steps and then follows an open-ended power schedule motivated by token-agnostic scheduling
work [ 7]:
ηbase(k) =
>><
>>:
5 × 10−3 k/1000, 0 ≤ k ≤ 1000,
5 × 10−4 + 4.5 × 10−3

1 + k − 1000
1000
−1/2
, k > 1000.
(5)
```

### Page 9

```text
Agnes 2B Pretraining Protocol Technical Report
The terminal Phase 1 base rate is approximately 1.04 × 10−3. Phase 2 contains approximately
152,586 steps and linearly decays the main trajectory from 1.04 × 10−3 to 10−5:
ηbase(s) = η0 + (ηmin − η0) s
S2
. (6)
Base LR ( ×10−3)
Cumulative tokens (B)
Phase 1
power decay
Phase 2
linear decay Final 29B
constant LR
0 438.8 900 1398.8
Figure 4 . Proposed base learning-rate schedule. The MuonH group uses ten times the plotted values.
5.4 Constant-Learning-Rate Branch and Model Averaging
For approximately the final 29B Phase 2 tokens, training resumes from a late checkpoint selected
by the physical non-padding token counter. The base learning rate is held near 4.08 × 10−5, giving
a MuonH effective rate of 4.08 × 10−4. Model weights are saved every 100 steps over approximately
the final 3.15B tokens. Six terminal checkpoints are averaged with equal weight:
¯θ = 1
6
6X
j=1
θj. (7)
Only model parameters are averaged; optimizer state, scheduler state, and data cursors are not.
Agnes does not hard-code global step identifiers from the reference study. The branch point and
averaging window are defined by frozen token ledgers and the non-padding token counter, avoiding
discrepancies caused by padding, partial batches, or incompatible counting conventions.
6 Training System and Cluster Topology
6.1 Node and Software Baseline
Each node contains eight 32GB RTX 5090 GPUs, with four GPUs associated with each CPU
socket, dual-socket x86 processors, at least 512GB of system memory, at least 8TB of local NVMe
storage, and one dual-port ConnectX-7 network adapter. Inter-node communication uses 400Gbps
InfiniBand. The production manifest freezes the exact Linux, NVIDIA driver, CUDA, NCCL,
Megatron Core [ 3], and Transformer Engine [ 4] versions together with the container digest.
Production runs use vendor-supported drivers and communication paths by default. The ref-
erence implementation relied on unsupported P2P/GDR modifications that cannot be disclosed
or reproduced completely; Agnes therefore excludes them from the reproducibility baseline. Any
experimental communication optimization must first pass compliance, security, and operations
review and must retain a stock-driver baseline, a rollback script, and a separately labeled throughput
report.
```

### Page 10

```text
Agnes 2B Pretraining Protocol Technical Report
6.2 Parallelism and Pipeline Balance
Because RTX 5090 devices lack NVLink, tensor parallelism would introduce expensive collectives at
every layer. Both phases therefore use TP=1 and combine pipeline parallelism with data parallelism,
as shown in Table 6.
T able 6. Production parallelism configurations and system gates.
Phase GPUs TP / PP / DP Pipeline layers Reported rate Agnes gate
Phase 1 24 1 / 2 / 12 (18 | 10) 238 TFLOP/s/GPU ≥ 220 TFLOP/s/GPU
Phase 2 96 1 / 4 / 24 (9 | 9 | 9 | 1) 192 TFLOP/s/GPU ≥ 175 TFLOP/s/GPU
The pipeline uses a 1F1B schedule and pp-dp rank ordering so that ranks in one pipeline group
remain topologically close where possible. Micro-batch size is 2. Because the untied LM head
contains approximately 311M parameters and is not concealed by FP8 execution of the internal
layers, the last stage receives only one Transformer layer. MuonH tensors remain whole where
possible; Adam state is distributed to balance residual memory. Stage execution times must differ
by less than 8%, and every stage must retain at least 1.5GB of peak-memory headroom.
6.3 Checkpointing, Recovery, and Observability
The fault-tolerance protocol is as follows:
• Save a fully recoverable state every 1,000 steps, including model parameters, master weights,
optimizer, scheduler, random-number generators, data cursor, non-padding token counter, and
numerical scalers. Save lightweight model weights every 250 steps.
• Retain the three latest full checkpoints locally and two remote replicas. After remote object
creation, verify file-level SHA-256 hashes and reread a random sample of tensors.
• Record loss, gradient norm, effective learning rate, throughput, memory use, temperature, and
communication time every 10 steps; evaluate the fixed validation set every 500 steps.
• After node loss, restore the latest full state and compare loss and throughput against the
historical band for 100 steps. Roll back if deviation exceeds three standard deviations over the
matched historical window.
• Conduct a recovery drill every 24 hours. The recovery-point objective is at most 1,000 steps
and the recovery-time objective is at most two hours.
7 Compute, Cost, and Execution Schedule
7.1 Compute and Wall-Clock Estimate
For a dense Transformer with approximately 2.032B parameters and 1.3988T training tokens, the
customary order-of-magnitude estimate is
C ≈ 6N D ≈ 1.71 × 1022 FLOPs. (8)
The 6N D approximation is a consistency check rather than a substitute for framework-level FLOP
accounting. Measurements reported for the structurally matched production setup imply 6,009
GPU-hours and 10.43 days for Phase 1, followed by 16,505 GPU-hours and 7.16 days for Phase 2.
The aggregate is 22,514 GPU-hours and 17.59 days. Agnes adds a 15% execution allowance, yielding
approximately 20.2 days of pure training capacity, plus two to three days for stage gates, recovery
drills, and final averaging.
```

### Page 11

```text
Agnes 2B Pretraining Protocol Technical Report
T able 7. Production-training planning baseline. Values are not Agnes measurements.
Phase GPUs Tokens Wall time GPU-hours Equivalent cost
Phase 1 24 438.84B 10.43 days 6,009 USD 1.84K
Phase 2 96 959.99B 7.16 days 16,505 USD 5.05K
Production total – 1.3988T 17.59 days 22,514 USD 6.89K
15% rerun allowance – – 2.64 equivalent days 3,377 USD 1.03K
The accounting rate is approximately USD 0.306 per GPU-hour under a five-year depreciation-
plus-electricity convention; it is not a public-cloud market rate. A complete project budget must
additionally cover CPU preprocessing, data storage, proxy and scaling experiments, failed trials,
networking and facility costs, evaluation, release engineering, compliance, and personnel.
7.2 Accelerator Budget Envelope
T able 8. Proposed accelerator budget envelope, excluding labor and infrastructure procurement.
Budget item Allowance Accounting scope
Production run USD 6.89K Reference-system measurement; recalculate after
the Agnes 1,000-step systems trial
Production rerun allowance USD 1.03K 15% of the production run for node failures and
one local fallback
Tokenizer, proxy, and data
ablations
USD 1.0K–1.8K Determined by the number of conditions, seeds,
and proxy tokens
Precision, optimizer, and
system scaling
USD 1.0K–1.7K Selective 170M/330M/600M/1.09B/1.7B
experiments
Recommended accelerator
envelope
USD 9.9K–11.4K Excludes data storage, networking, power
equipment, infrastructure purchase, and labor
7.3 Eight-Week Execution Plan
W1 W2 W3 W4 W5 W6 W7 W8
Data + tokenizer
Scaling ladder
Phase 1
Phase 2
Acceptance + release
License + data + T0/D0
Precision + optimizer + systems
438.84B
959.99B
Audit + average + release
Figure 5 . Indicative eight-week schedule. Data work and system scaling overlap; the production run
begins only after all prerequisite gates pass. Bars denote planning windows, not measured completion dates.
```

### Page 12

```text
Agnes 2B Pretraining Protocol Technical Report
8 Evaluation Protocol and Release Gates
8.1 Fixed Evaluation Protocol
The primary base-capability suite contains 15 tasks: GSM8K, MATH, sanitized-MBPP, HumanEval,
MMLU, MMLU-Pro, ARC-Challenge, ARC-Easy, BoolQ, CommonsenseQA, HellaSwag, PIQA,
SocialIQA, WinoGrande, and BBH. Generative tasks use greedy decoding and a frozen parser.
Multiple-choice tasks report both perplexity-based and generated-answer metrics; the primary
convention is frozen before training and may not be selected retrospectively per model.
Chinese evaluation adds C-Eval, CMMLU, and an internal decontaminated Chinese knowledge-
and-reasoning set. Code evaluation adds multilingual unit tests. At the Base-model stage, long-
context testing is limited to consistency within 4K tokens. Results obtained by extrapolating RoPE
beyond the trained context do not contribute to the primary release score. Every benchmark freezes
its version, prompt template, shot count, maximum generation length, stop sequences, parser, and
container digest.
The aggregate labels Avg15, Math/Code Avg4 , and Reason/Knowledge Avg11 refer to metric
groups frozen in the evaluation manifest before the first production checkpoint is evaluated. The
manifest must enumerate group membership and normalization rules; the labels alone are insuﬀicient
to redefine a group after scores are observed.
8.2 Reference Performance and Agnes Target Profile
Table 9 records the final base-model scores reported by the reference study under its common
OpenCompass pipeline [ 1]. These percentages are external performance anchors : they are
neither Agnes measurements nor forecasts. Individual task values remain diagnostic because small
evaluation or tokenizer changes can move a single score materially. Agnes preregisters hard floors
only for validation loss and the three aggregates, with stretch goals used to rank checkpoints that
already pass every release gate.
The reference pipeline uses greedy generation for GSM8K, MATH, sanitized-MBPP, HumanEval,
MMLU-Pro, and BBH, and perplexity ranking for the other nine tasks. For applicable multiple-
choice tasks, the reference aggregate selects the better cloze or multiple-choice formulation indepen-
dently for each model. Agnes instead freezes one primary formulation before training. Consequently,
a claim of numerical parity requires a protocol-matched rerun; the tabulated values alone do not
establish parity.
T able 9. Reference-reported performance anchors and their preregistered Agnes use. All capability scores
are percentages.
Indicator Reference protocol Reported anchor Agnes preregistered use
Phase 1 validation loss Held-out next-token
loss
2.730 G4 floor ≤ 2.75; stretch ≤ 2.73
Phase 2 trajectory loss Held-out next-token
loss
2.488 G5 candidate floor ≤ 2.52;
measure the averaged export
separately
GSM8K GEN, 4-shot 59.67 Diagnostic only; no task-level
floor
MATH GEN, 4-shot 30.30 Diagnostic only; no task-level
floor
sanitized-MBPP GEN, 3-shot 52.92 Diagnostic only; no task-level
floor
HumanEval GEN, 3-shot 31.10 Diagnostic only; no task-level
floor
```

### Page 13

```text
Agnes 2B Pretraining Protocol Technical Report
T able 9. Reference-reported performance anchors and their preregistered Agnes use (continued).
Indicator Reference protocol Reported anchor Agnes preregistered use
Math/Code A vg4 Unweighted mean of
four tasks
43.50 G5 floor ≥ 42.0; stretch ≥ 43.5
MMLU PPL, 5-shot 57.44 Diagnostic only; no task-level
floor
MMLU-Pro GEN, 5-shot CoT 25.81 Diagnostic only; no task-level
floor
ARC-Challenge PPL, 5-shot 74.24 Diagnostic only; no task-level
floor
ARC-Easy PPL, 5-shot 86.07 Diagnostic only; no task-level
floor
BoolQ PPL, 5-shot 76.85 Diagnostic only; no task-level
floor
CommonsenseQA PPL, 5-shot 67.81 Diagnostic only; no task-level
floor
HellaSwag PPL, 5-shot 63.21 Diagnostic only; no task-level
floor
PIQA PPL, 5-shot 75.52 Diagnostic only; no task-level
floor
SocialIQA PPL, 5-shot 63.61 Diagnostic only; no task-level
floor
WinoGrande PPL, 5-shot 60.62 Diagnostic only; no task-level
floor
BBH GEN, 4-shot 42.05 Diagnostic only; no task-level
floor
Reason/Knowledge
A vg11 Unweighted mean of
eleven tasks
63.02 G5 floor ≥ 62.5; stretch ≥ 63.0
A vg15 Unweighted mean of
all tasks
57.81 G5 floor ≥ 57.0; stretch ≥ 57.8
Chinese extension C-Eval, CMMLU,
internal set
– Freeze numeric thresholds
after 600M calibration; report
separately from A vg15
The reference study does not report Chinese-specific base-model scores. Agnes therefore does not
invent an absolute Chinese target from unrelated evidence. The 600M calibration run must freeze
the C-Eval, CMMLU, and internal-set thresholds before the 1.7B scaling run, while G4 retains a
no-material-regression condition throughout production.
8.3 Stage Gates
T able 10. Agnes pretraining and release gates.
Gate Time Pass criterion Action on failure
G0 Data and
pipeline
Tokenizer and shard hashes
frozen; 100-step replay loss
difference < 10−5; no exact
benchmark-data match under
the frozen contamination
protocol
Block scale-up and roll
back the data or packing
version
```

### Page 14

```text
Agnes 2B Pretraining Protocol Technical Report
T able 10. Agnes pretraining and release gates (continued).
Gate Time Pass criterion Action on failure
G1 FP8 numerical
path
At 170M and 600M,
FP8–BF16 validation-loss
difference ≤ 0.005; throughput
≥ 1.25×; no NaN or Inf
Return sensitive layers to
BF16; recalibrate scaling
and initialization
G2 Optimizer Two sizes and at least two
seeds; MuonH endpoint no
worse than the tuned baseline;
no anomalous effective-LR
collapse
Retain AdamW or standard
Muon as the production
fallback
G3 Systems 24-GPU rate ≥ 220 and
96-GPU rate ≥ 175
TFLOP/s/GPU; stage
imbalance < 8%; memory
headroom ≥ 1.5GB
Adjust PP layout,
MBS, and overlap; then
recompute time and budget
G4 Phase 1 Validation loss ≤ 2.75; curve
no more than 2% outside the
600M extrapolation band;
no statistically material
regression on Chinese or
general proxies under the
frozen test
Pause Phase 2 and inspect
data weights, LR, and
numerical execution
G5 Final capability Validation loss ≤ 2.52; A vg15
≥ 57.0; Math/Code A vg4
≥ 42.0; Reason/Knowledge
A vg11≥ 62.5; see Table 9
Compare uniform,
curriculum, and averaged
endpoints; select only a
checkpoint that passes
G6 Release Decontamination
report, license manifest,
reproducibility checklist,
weight hashes, and model
card approved; no P0 safety
blocker
Do not release the failing
checkpoint; retain an
internal audit copy
Origin and status of the target values
The loss and aggregate capability thresholds in G4 and G5 are derived from endpoints reported
at a similar model scale and training-token count, with a small engineering tolerance. They
are Agnes acceptance targets , not performance promises. Per-task anchors in Table 9 are
diagnostic and do not create hidden release gates. If tokenizer choice, licensing constraints, or
stock-driver throughput changes the experimental conditions, reviewers may revise a threshold
before production starts. Every revision must be versioned; no gate may be rewritten after
final scores are observed.
```

### Page 15

```text
Agnes 2B Pretraining Protocol Technical Report
8.4 Minimum Ablation Matrix
T able 11. Minimum experimental matrix before and during production.
Experiment Scale / budget Control Primary criteria
Precision pairing 170M, 600M; 20
TPP
BF16 vs. blockwise FP8; same seed Loss delta,
throughput,
saturation, memory
Optimizer pairing 170M, 600M;
two seeds
AdamW, tuned Muon, MuonH Endpoint loss,
effective LR, stability
LR schedule 600M; 20/50
TPP
Grid over peak rate and decay ratio Endpoint loss,
extrapolation error
Systems scaling 1.7B; 1,000
steps
8/24/96 GPUs; candidate PP layouts TFLOP/s, stage
balance, AllReduce
time
Data proxy 0.6B; source
slices
Shared checkpoint plus replay Four-axis vector,
variance
Curriculum order 2B; Phase 2
shadow
Uniform vs. curriculum; identical tokens A vg15,
contamination, loss
Late branch 2B; final 29B Decay final/average; constant-LR average Stability,
multi-endpoint
capability
Curriculum ordering, constant learning rate, and model averaging do not have full-factorial evidence.
The final report must therefore avoid attributing a joint improvement to any single component
without a corresponding control. Under budget pressure, the uniform shadow and the three terminal
comparators are retained because they determine whether to release the curriculum-averaged model
or a more conservative single checkpoint.
9 Risk Analysis, Limitations, and Stop Conditions
9.1 Risk Register
T able 12. Principal technical and governance risks.
ID Risk Early signal Prevention and response
R1 FP8 numerical
divergence
Rising amax or saturation,
loss spikes, non-finite
gradients
Run paired BF16 controls;
whitelist sensitive layers;
pause automatically; restore
a healthy checkpoint and fall
back layerwise
R2 Consumer-GPU
communication
bottleneck
96-GPU rate below 175
TFLOP/s/GPU; increasing
synchronization fraction
Use TP=1, topology-aware
ranks, and asymmetric PP;
retain stock drivers as the
reproducible baseline
R3 Lack of ECC or
node failure
GPU Xid events, silent
corruption, temperature or
power drift
Hold 10% spare GPUs;
sample tensor hashes; exercise
RPO/RTO procedures; isolate
failing nodes automatically
```

### Page 16

```text
Agnes 2B Pretraining Protocol Technical Report
T able 12. Principal technical and governance risks (continued).
ID Risk Early signal Prevention and response
R4 Licensing or PII
failure
Missing license, withdrawal
request, anomalous PII hit
rate
Enforce license whitelist and
quarantine; retain deletable
provenance; require legal
approval before release
R5 Apparent
curriculum
gain caused by
contamination
Rising near-duplicate rate
between late buckets and
benchmarks
Apply exact, MinHash, and
embedding-based benchmark
audits; report decontaminated
reevaluation
R6 Misinterpreted
source-local score
Higher-score buckets reduce
proxy capability
Validate the semantic
direction of every source score;
otherwise use a fixed random
order
R7 Token-ledger
inconsistency
Materialized, step-derived,
and non-padding counts
differ by more than 0.1%
Freeze tokenizer first; maintain
three ledgers; define late
averaging by token count
rather than step ID
R8 Proxy decisions fail
to extrapolate
Model-size ordering
changes or variance across
seeds is high
Confirm critical choices at two
scales; retain the conservative
uniform alternative when
disagreement persists
R9 Cost scope is
misrepresented
USD 6.89K is described as
total project cost
Separate accelerator-only,
infrastructure, and labor
accounting in every external
document
9.2 Mandatory Stop Conditions
Training stops if any of the following occurs: validation loss worsens at two consecutive full validation
points after excluding a scheduled data-transition effect; a non-finite parameter or gradient cannot
be cleared by the automatic fallback path; licensing review identifies a P0 blocker; two consecutive
checkpoints fail to restore; measured throughput would exceed the approved budget by more
than 15% without a signed change request; or evaluation contamination cannot be repaired while
preserving provenance.
9.3 Limitations
The plan has several material limitations. First, the principal eﬀiciency evidence comes from shorter
token-per-parameter regimes than the proposed production run; optimizer and curriculum rankings
may change at approximately 700 tokens per parameter. Second, consumer GPUs lack ECC and
NVLink, creating reliability and communication risks that accelerator-hour accounting alone does
not capture. Third, the production tokenizer and several optimizer details are original Agnes
choices rather than disclosed facts from the reference study. Fourth, the initial contamination policy
cannot guarantee removal of semantic paraphrases or undisclosed benchmark derivatives. Fifth, the
40,960-token structural RoPE setting is not evidence of long-context training or performance; release
evaluation remains capped at 4K. Finally, the proposed cost excludes material categories of project
expenditure and depends on reproducing the measured throughput with supported software.
These limitations motivate the gate structure. Failure of an individual method does not require
abandoning the program: BF16, AdamW or standard Muon, uniform sample order, stock drivers,
```

### Page 17

```text
Agnes 2B Pretraining Protocol Technical Report
and single-checkpoint release remain conservative fallbacks. A fallback changes time, cost, or quality
expectations and therefore requires a versioned planning update.
10 Reproducibility and Release Artifacts
10.1 Required Deliverables
The release candidate must contain:
1. Agnes 2B Base: BF16 SafeTensors, model configuration, generation configuration, and file-
level SHA-256 hashes.
2. Agnes tokenizer: model files, special-token map, normalizer, tokenizer-training manifest, and
fertility report.
3. Data package: source-and-license manifest, processing configuration, deduplication and de-
contamination reports, and phase- and bucket-level token ledgers. Restricted data are repre-
sented by reconstructable manifests and scripts rather than redistributed content.
4. T raining package: container digest, Megatron configuration, parallel layouts, random seeds,
full logs, key checkpoints, and recovery instructions.
5. Evaluation package: dataset versions, prompts, shot counts, decoding parameters, parsers,
per-example outputs, and aggregation scripts.
6. Model card and technical report: intended scope, known limitations, cost boundary, data
risks, base capabilities, and interfaces for subsequent training.
10.2 Pre-Release Checklist
T able 13. Reproducibility checklist. Every item remains pending until supported by project evidence.
Status Item Required evidence
□ Raw-data versions and licenses
locked
Source URL, version, license, access date,
redistribution rule, and hashes
□ Tokenizer fully frozen Training manifest, vocabulary, normalizer,
special tokens, and round-trip report
□ Cleaning, deduplication,
and contamination checks
reproducible
Code revision, configuration, seeds,
thresholds, output hashes, and deletion
map
□ Three token ledgers reconciled Unique documents, materialized tokens,
and physical non-padding tokens
□ Training environment locked Container, driver, CUDA, NCCL,
framework, and kernel versions
□ Model and optimizer
configuration complete
Architecture, initialization, parameter
groups, LR, precision, parallelism, and
exact parameter count
□ Checkpoint restoration
demonstrated
Model, master weights, optimizer,
scheduler, RNG, data cursor, and
recovery-drill logs
□ Evaluation version locked Dataset versions, prompts, decoding,
parsers, per-example outputs, and
statistics scripts
```

### Page 18

```text
Agnes 2B Pretraining Protocol Technical Report
T able 13. Reproducibility checklist (continued).
Status Item Required evidence
□ Cost accounting auditable Active GPU-hours, downtime,
experiment/production split, rate
assumptions, and exclusions
□ Final artifacts verifiable Weights and configuration hashes, model
card, license manifest, and approval
record
11 Conclusion
This report defines an auditable route to pretraining Agnes 2B rather than merely reproducing a
low accelerator-cost headline. The plan combines consumer-GPU execution, blockwise FP8, explic-
itly scaled optimizer groups, source-local curriculum ordering, and late-stage parameter averaging
within a single 1.3988T-token ledger. It decomposes the program into tokenizer and data freezes,
scale-model experiments, two production phases, a uniform shadow branch, and multi-endpoint
evaluation. It also closes common planning gaps in licensing, decontamination, recovery, and
total-cost interpretation.
The performance profile is explicit and falsifiable. The final checkpoint must reach validation loss
at most 2.52, A vg15 at least 57.0, Math/Code A vg4 at least 42.0, and Reason/Knowledge A vg11 at
least 62.5 under the frozen Agnes protocol. The corresponding external capability anchors are 57.81,
43.50, and 63.02; the separate Phase 2 trajectory-loss anchor is 2.488. All individual benchmark
values remain diagnostics rather than undisclosed release conditions.
The recommended decision is to authorize T0/D0 and the 170M–1.7B scaling ladder first. The 24-
to-96-GPU production window should be committed only after G0–G3 pass. This structure ensures
that if an externally reported method does not reproduce on Agnes data or under a supported
driver, the team can adopt a validated conservative fallback before consuming the full 1.4T-token
budget.
```

### Page 19

```text
Agnes 2B Pretraining Protocol Technical Report
A Frozen Configuration Summary
T able 14. One-page summary of the proposed production configuration.
Item Frozen value or rule
Architecture 28 layers, d = 2048 , df f = 6144 , Q/KV heads=16/8, head dimension=128,
untied LM head
Tokenizer Agnes byte-level BPE, V = 151,936; version and hashes frozen at T0
Sequence / batch 4,096 / GBS 1,536 / MBS 2; at most 6,291,456 physical tokens per step
Phase 1 438.84B tokens; 24 GPUs; TP/PP/DP=1/2/12; power LR; layout (18 | 10)
Phase 2 959.99B tokens; 96 GPUs; TP/PP/DP=1/4/24; linear LR; layout (9 | 9 | 9 | 1)
Precision Linear E4M3 FP8; activation groups 1×128; weight blocks 128×128; remaining
paths BF16/FP32
Optimizer MuonH for selected matrices, WD 0 and 10× base LR; AdamW for remaining
parameters, WD 0.1
Curriculum 43.87B-token transition; 376 source-local progress buckets; same-token uniform
shadow
Late branch Constant LR over approximately the final 29B tokens; equal model-parameter
average of six terminal checkpoints
Checkpointing Full state every 1,000 steps; lightweight model every 250; RPO ≤ 1,000 steps
Performance Final loss ≤ 2.52; A vg15 ≥ 57.0; Math/Code A vg4 ≥ 42.0; Reason/Knowledge
A vg11≥ 62.5
Release All G0–G6 gates pass; BF16 Base, tokenizer, manifests, code, logs, evaluation,
and model card
B Separation of Reported Facts and Agnes Planning Values
T able 15. Final check against presenting planned values as achieved results.
Quantity Epistemic status Permitted use
238 / 192 TFLOP/s/GPU Reported by the
reference study
Systems anchors only; Agnes release gates are
220 / 175
2.730 / 2.488 validation loss Reported by the
reference study
Basis for G4/G5 targets; never claim Agnes has
reached them
22,514 GPU-hours / USD 6.89K Reported under a narrow
accounting scope
Production anchor; experimental and rerun
budgets remain separate
FP8 loss difference
0.0031–0.0039
Reported scaling
evidence
Agnes revalidation threshold is at most 0.005
A vg4 43.50 / A vg11 63.02 /
A vg15 57.81
Reported endpoints Agnes G5 floors are 42.0 / 62.5 / 57.0; individual
scores remain diagnostic
2.032B configuration, tokenizer,
and recovery policy
Agnes design decisions Generate project evidence through tensor export,
T0, and recovery drills
C Token and Step Accounting
Table 16 records the accounting identities used by the scheduler and the release ledger. Displayed
domain components are rounded to two decimals and displayed shares to one decimal; phase totals,
rather than sums of rounded cells, are authoritative.
```

### Page 20

```text
Agnes 2B Pretraining Protocol Technical Report
T able 16. Authoritative token-accounting identities.
Quantity Value Interpretation
Maximum tokens per step 6,291,456 1,536 sequences × 4,096 tokens
Phase 1 authoritative total 438.84B Approximately 69,752 optimization steps
Phase 2 component pool 938.06B Unique pool before Phase 1 replay
Phase 1 replay in Phase 2 Approximately 21.93B Integral of the declining replay weight over
the 43.87B transition
Phase 2 physical total 959.99B Pool consumption plus replay
Full-run total 1.3988T Phase 1 plus physical Phase 2 training
tokens
Phase 2 steps Approximately 152,586 Reconciled at runtime using the
non-padding token counter
References
[1] K. Luo et al., “Low-cost open pretraining on consumer GPUs,” arXiv preprint arXiv:2608.27370 , 2026.
Reference study used for the reported planning anchors in this report.
[2] Qwen Team, “Qwen3-1.7B model configuration,” oﬀicial model repository, 2025.
[3] NVIDIA, “Megatron Core v0.16.0,” oﬀicial release and documentation, 2026.
[4] NVIDIA, “Transformer Engine: FP8 and FP4 training guide,” version 2.17 documentation, 2026.
[5] K. Wen, X. Dang, K. Lyu, T. Ma, and P. Liang, “Fantastic Pretraining Optimizers and Where to Find
Them II: Hyperball Optimization,” arXiv preprint arXiv:2606.16899 , 2026.
[6] K. Luo et al., “How Learning Rate Decay Wastes Your Best Data in Curriculum-Based LLM Pretraining,”
International Conference on Learning Representations , 2026.
[7] Y. Shen et al., “Power Scheduler: A Batch Size and Token Number Agnostic Learning Rate Scheduler,”
arXiv preprint arXiv:2408.13359 , 2024.
```

