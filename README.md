# lawlike

**Computable toy experiments on a non-computable question: under a simplicity prior, is law-like experience typical rather than miraculous, and can "observer" be defined without an arbitrary choice?**

This repository turns a line of argument into experiments you can run on a laptop. The first experiment (elementary cellular automata) is complete and runnable; the roadmap below says what comes after it and why.

---

## 1. The argument being tested

1. **Boltzmann brain problem.** If every arrangement of matter or experience is equally likely, a typical observer should see noise, not stable laws. Our experience is stable and law-like, so either we're astronomically lucky or the measure isn't uniform.
2. **Müller's answer** (*Law without law*, Quantum 2020). Put the measure on observer states directly and weight continuations by algorithmic probability, roughly 2^-(length of shortest program). Then, provably and asymptotically, observers tend to experience stable, simple regularities, as if they were embedded in an external, computable world. The proofs work without computing anything: they're counting arguments about short programs dominating the prior mass.
3. **The catch: it's uncomputable.** Solomonoff's prior can't be evaluated. But computable stand-ins exist: resource-bounded priors, sequential mixture predictors, and compressors. That's what this repo uses.
4. **The observer is a program too.** "World + a rule that reads an experience out of it" is just one longer program. If observers get weight 2^-L like worlds, summing over (world, observer) pairs is marginalisation, not an arbitrary choice. The hypergraph-or-CA choice survives only as a choice of reference machine.
5. **The trivial-observer catch.** Under a bare 2^-L measure, the cheapest "observers" ignore the world and emit a constant or periodic string. They're perfectly lawful and perfectly meaningless. So we need structural criteria for what counts as an observer, and the criteria must be justified independently, not tuned to give the answer we want.
6. **Candidate criteria.** Each comes from a different research tradition: Crutchfield's local causal states, Krakauer et al.'s information theory of individuality, Friston's Markov blankets, IIT-style integration and exclusion, and Wolfram-style computationally bounded, persistent observers.
7. **Laws as invariants.** A regularity that holds for most admissible observer-mass, and survives perturbations of the facts, is a non-arbitrary candidate for what a "law" is.

The experiments ask, at each step, whether these claims survive contact with an actual computable system, and where they break.

---

## 2. Roadmap

Each phase has a question, a justification, and an exit criterion, so we know when to move on and when to stop.

### Phase 0 — Instrument checks (done)
`python -m lawlike.selftest`. Every estimator is tested on cases with a known answer: noise is noisy, periodic is lawful, independent variables have zero mutual information, rule 0 has one causal state, and so on.
*Why first:* if the microscope is broken, nothing we see through it means anything.
*Exit:* all checks pass. Re-run after any change to `info.py`, `causal.py`, or `observers.py`.

### Phase 1 — World-first, elementary CAs (done)
*Question:* with worlds weighted by a simplicity prior and observers weighted by their own description length, does admissible observer-mass mostly experience law-like streams? Is that more than under a uniform (Boltzmann) prior or pure noise? Do the different observer criteria agree on who the observers are?
*Why ECAs:* 256 rules (88 up to symmetry), cheap to simulate, and well studied. Gliders, domains, and chaos are all present and all have known causal-state structure. It's the smallest substrate where every piece of the argument can be instantiated.
*Exit:* hypotheses H1–H5 (§3.6) each get a verdict on the `full` preset, and the filter-health table has told us which criteria to carry forward. *Reached 2026-10-23;* verdicts and the decisions they led to are in `NOTES.md` and `results/full/report.md`.

### Phase 2 — Let the prior choose the dynamics, not just the initial condition
In Phase 1 every rule costs exactly 8 bits, so the prior only acts on initial conditions. Next: totalistic and outer-totalistic families, radius-2 and 3-colour CAs, and string-rewriting / substitution systems, with **variable-length rule encodings**, so simpler rules really are cheaper. Split into 2a–2d so the substrate and the instruments never change in the same step; the design is pre-registered in §3b.
*Why:* Müller's claim is about simple programs, and the program includes the dynamics. String rewriting is also the 1-D ancestor of Wolfram's hypergraph models, so it's the bridge to Phase 5.
*Exit:* the Phase 1 verdicts either replicate or change as rule-space grows. Whichever happens is the result.

### Phase 3 — Observer-first (the Müller arm)
Drop the world entirely. Generate experience streams directly from a speed prior over short programs in a tiny universal language (a minimal register machine or tag system), the closest computable analogue of Müller's setup. Then ask whether those streams look *as if* generated by a world: can a small CA plus an observer reproduce them, and are they lawful in the Phase 1 sense?
*Why:* Phase 1 assumes a world and asks about observers in it; Müller derives the world from the observer. If both directions agree, that's evidence the picture is coherent rather than an artefact of the world-first framing.

### Phase 4 — Invariance and renormalisation
Treat coarse-grainings as a family; for each world, find regularities (learned predictive models) that hold across most admissible observer-mass. Define a "law" as such an invariant and track whether the set of invariants **stabilises or fragments** as substrates get richer.
*Why:* this is the non-arbitrary definition of law from the discussion. Phase 1's H4 (facts change, laws hold) is the first, crude version of it.

### Phase 5 — Hypergraph rewriting
Enumerate small Wolfram-model rules, weight them 2^-L, build causal graphs, and port causal states and individuality to graph neighbourhoods.
*Why last:* simulating hypergraphs is expensive, and the machinery should be debugged on CAs and strings first. The substrate choice matters only through bounded constants (invariance theorem). The interesting question is whether results from cheaper substrates carry over.

### Phase 6 — Continuers (the quantum-immortality question, made computable)
Take a law-like observer stream prefix. Under a speed prior, enumerate its cheapest continuations, then measure the trade-off between a continuation's weight and how connected it stays to the prefix: does it keep the laws, keep the observer's structure, or dissolve into a cheap flicker?
*Why:* this is the amplitude-versus-connectedness curve from the original discussion, measured rather than argued. It's where "the cheapest continuer wins, and it isn't you" becomes a number.

---

## 3. Experiment 1 design

### 3.1 Worlds and their priors
A world is a program: (ECA rule, initial condition). Three arms:

| arm | initial condition | weight | role |
|---|---|---|---|
| **simplicity** | a seed of length ℓ, either tiled across the lattice or embedded in zeros | 2^-(1 + EliasGamma(ℓ) + ℓ), summed over every program giving the same world | the computable stand-in for Solomonoff/Müller |
| **uniform** | a uniformly random N-bit string | equal | the Boltzmann measure: every microstate alike |
| **noise** | none; every spacetime cell is a fair coin | equal | the no-law baseline |

Rules are the 88 symmetry classes, weighted by class size. Each rule costs the same 8 bits, so in Phase 1 the prior acts only through the initial condition (see Phase 2). Seeds up to `ic_exact_len` are enumerated exactly; longer ones are sampled and importance-weighted. Programs that produce identical worlds (tile `0` = tile `00` = embed `0`) have their weights **summed**, as in Solomonoff's definition.

Each world is also run with **one cell flipped** halfway through, for the perturbation test.

### 3.2 Observers (each one a program with a description length)

| family | reads | description length L (bits) |
|---|---|---|
| **null** | nothing — emits a fixed periodic pattern | 2 + EG(period) + period |
| **window** | w raw cells, drifting at speed p/q (it can ride a glider) | 2 + EG(w) + log2(#speeds) + log2(N) |
| **coarse** | w blocks of s cells, majority-voted (a bounded observer) | window + EG(s) |
| **causal** | w local causal-state labels | window + EG(hp) + EG(hf) |

Prior weight 2^-L. Start positions are sampled on a stride grid and importance-weighted, so a family's total mass doesn't depend on N. The **null family is a deliberate control**: it's the trivial observer from §1.5, and any criterion worth keeping has to exclude it for a reason.

### 3.3 Admission criteria ("filters")
Each filter scores every observer; the top `top_q` (20%) with a meaningfully positive score are admitted. Every filter is run on every world, so the criteria can be compared directly.

| filter | source | measures | known weakness |
|---|---|---|---|
| `none` | — | admits everything (bare 2^-L) | trivial observers |
| `persistence` | Friston (self-evidencing), Wolfram (persistent observers) | I(S_t; S_t+1) / H(S): does its own state predict its next state? | periodic null observers pass; it can't tell world from no world |
| `individuality` | Krakauer et al. 2020 | I(S_t+1; S_t \| E_t): self-prediction *beyond* what the environment explains | plug-in estimates at short T |
| `blanket` | Friston; Markov blankets | 1 − leak, where leak = I(S_t+2; outside_t \| S_t, boundary_t), normalised | blankets are easy to find (Biehl et al. 2021): frozen regions are perfect blankets |
| `integration` | IIT-flavoured (Ay's stochastic interaction) | min over bipartitions of whole-minus-parts temporal mutual information | not IIT 3.0/4.0 phi; zero for single cells by construction |
| `integration_excl` | IIT exclusion postulate | among overlapping candidates (same start and drift), only the maximally integrated grain exists | inherits `integration`'s weaknesses |
| `causal_complexity` | Shalizi; Rupe & Crutchfield 2018 | mean local statistical complexity under the footprint: is it on a coherent structure rather than in the background domain? | finite light cones; merge threshold |

### 3.4 Law-likeness of an experience stream
Each observer's stream is coded by a **Bayesian mixture of Markov models of order 0–8 with KT estimators**. This is a sequential predictor, and it's the computable stand-in for "how surprised would a simplicity-prior predictor be?"

- **H1** = marginal entropy (variety); **h** = predictor's code length per symbol (surprise); **Λ = H1 − h** = predictable variety.
- Classes: **trivial** if H1 < 0.1 bits; **lawful** if h/H1 ≤ 0.5; otherwise **noisy**.
- **Law drift:** the second half coded by the model frozen after the first half, compared with continued learning.
- **Facts vs laws:** after the one-cell flip, did later experiences change (divergence > 5%), and does the model learned *before* the flip still predict the changed future (robust gap < 0.1 bits/symbol)?
- **Intersubjectivity:** normalised compression distance between streams of admitted observers in the same world, compared against random raw windows.

### 3.5 Anthropic weighting — both, always
- **SSA-style:** each world counts by P(world); its admitted observers share that weight.
- **SIA-style:** each (world, observer) pair counts by P(world)·P(observer), so worlds hosting more observer-mass count more.

These are the two standard choices and neither is obviously right, so both are reported everywhere. Where they disagree, the disagreement is itself a result.

### 3.6 Pre-registered hypotheses
Written before any full run. Thresholds are in `experiment1.py` and `config.py` and are **not** to be tuned on the run being evaluated.

- **H1 — simplicity makes lawfulness typical.** lawful share (simplicity) > (uniform) > (noise), for most world-reading filters. *Falsified if* the simplicity arm is no more lawful than the uniform arm under the filters that pass the health checks.
- **H2 — the trivial-observer catch is real, and filters fix it.** Under `none`, null observers and frozen streams take a large share of mass. World-reading filters should drive the null share to zero and keep the trivial share low.
- **H3 — criteria converge.** Independently motivated filters admit overlapping observers, at better than chance (~0.11 Jaccard for two random top-20% sets). *If they don't converge*, "observer" may not be one natural kind at this scale. That would be a substantive finding, not a bug.
- **H4 — facts change, laws hold.** In the simplicity arm, a one-cell flip often changes admitted observers' later experience while leaving their learned laws predictive.
- **H5 — intersubjectivity.** Admitted observers in a world share more information with each other than random windows do (lower NCD). This is the toy version of Müller's "observers come to agree on a common world."

---

## 3b. Phase 2 design (pre-registered 2026-10-23, before any Phase 2 result)

Written after Phase 1 `full` and before computing anything under a new rule prior, including the reweighting preview (§3b.6), whose data already exists. Changing anything here after seeing Phase 2 results follows §5.1.

### 3b.1 Sub-phases
One thing changes per step, so a change in the verdicts can be pinned to a cause.

| step | substrate | what changes | instruments |
|---|---|---|---|
| **2a** | binary, radius 1 (the same 256 ECAs) | only the prior over rules | unchanged |
| **2b** | binary, radius 1 and 2; general, totalistic, outer-totalistic | rule space (2³² general radius-2 rules, sampled) | light cones and observer boundary width generalised to radius r, with new self-tests, *before* any radius-2 world is run |
| **2c** | k colours | alphabet | binary assumptions in observers, causal states and coarse-graining generalised first |
| **2d** | string rewriting / substitution systems | growing, non-lattice worlds | a new mapping from observer to world; designed separately |

At radius 1, every totalistic and outer-totalistic binary rule is an ECA, so 2a runs exactly the Phase 1 worlds and isolates the effect of the prior over dynamics.

### 3b.2 Rule priors (reference machines)
The encoding is where our bias enters most directly. The invariance theorem promises only that results under two machines differ by a bounded constant; we test whether that constant matters here by running two machines. **A conclusion that holds under only one machine is reported that way.**

**M_A (primary): family table.** program = 2-bit family tag (totalistic / outer-totalistic / general; one code unused) + EliasGamma(r) + rule table. Binary table sizes: totalistic 2r+2 bits (one output per neighbourhood sum), outer-totalistic 2(2r+1) bits (centre × sum of the others), general 2^(2r+1) bits. At r = 1 the programs are 7, 9 and 11 bits long. As in Phase 1, a rule's weight is the **sum** over every program that produces it: totalistic ⊂ outer-totalistic ⊂ general, so a totalistic ECA gets 2⁻⁷ + 2⁻⁹ + 2⁻¹¹ (21× a general-only rule) and an outer-totalistic one 2⁻⁹ + 2⁻¹¹ (5×). Rule-weight entropy: 6.94 bits (Phase 1: 8). For 2c, EliasGamma(k−1) is inserted after the tag; 2c may revise this before it runs.

**M_B (sensitivity, 2a only): shortest Boolean formula.** A rule is a formula over its neighbourhood in prefix notation, from 10 tokens {NOT, AND, OR, XOR, XNOR, l, c, r, 0, 1}, each costing log₂10 bits; weight ∝ 10^−(minimal formula size). The alphabet is closed under De Morgan duality and l↔r, so all four members of an ECA symmetry class get the same weight. M_B differs from M_A in two ways at once, the machine and min-versus-sum; that is deliberate, since a sensitivity arm should be sensibly different, not a small perturbation. Its known character, computed from the rule weights alone: it is steep (rule-weight entropy 2.82 bits), with 92% of its weight on five dynamically trivial rules (0, 255, the identity 204 and the shifts 170 and 240). Minimal sizes cannot be enumerated for the 2³² radius-2 rules, so 2b's sensitivity machine is pre-registered before 2b.

The Phase 1 weighting (every rule 8 bits) is reported next to both as the reference point.

### 3b.3 Arms
| arm | rule | initial condition | role |
|---|---|---|---|
| simplicity | M_A (or M_B) | seed prior (§3.1) | the Müller stand-in, now including the dynamics |
| uniform-IC | M_A (or M_B) | uniform random | isolates the prior over initial conditions |
| uniform-program | uniform over rules | uniform random | the full Boltzmann measure; at r = 1 this is Phase 1's uniform arm |
| noise | — | — | no-law baseline |

### 3b.4 Primary analysis (fixed now)
- **Filters counted toward verdicts:** individuality, integration, integration_excl, causal_complexity. `persistence` and `blanket` are deprecated (they failed filter-health checks (a) and (b) on Phase 1 `full`). They are still computed and reported, marked †. "Most filters" means **at least 3 of these 4**.
- **Co-primary analyses:** all worlds, and rich worlds (spatial 3-block entropy ≥ 0.5 bits/cell, fixed from Phase 1). If they disagree, both are reported and neither is the headline.
- **Weighting:** verdicts use SSA; SIA is reported everywhere. A conclusion that holds under only one is stated that way.
- **Intervals:** the two-way cluster bootstrap in `bootstrap.py`, which resamples rules and sampled seeds as crossed factors (rules resampled: the superpopulation view). Simplicity − uniform contrasts are paired on rules.
- **H3** is tested as the mean pairwise overlap of the four counted filters against chance (0.111). The version over all filters, as pre-registered in §3.6, is also reported.
- **H5** is replaced (see NOTES.md for why the NCD version failed): *aligned mutual information* between admitted, world-reading observers whose footprints never overlap, normalised by the smaller entropy and debiased by permutation, minus the same statistic for random raw windows. That gives a gap per world. **Verdict on simplicity gap − noise gap**: a statistic that doesn't separate a structured world from noise isn't testing intersubjectivity. If a filter admits no pairs of world-reading observers in any noise world, its noise gap counts as 0 (there is no common world to agree about), and the verdict is on the simplicity gap alone. The NCD version stays as a secondary statistic.

### 3b.5 Hypotheses for 2a
Each one gets a verdict under M_A (primary) and M_B (sensitivity), in both co-primary analyses.
- **H1–H5:** as in §3.6, with the amendments in §3b.4.
- **H6 — dead worlds grow.** Under M_A, the share of simplicity-arm weight in non-rich worlds rises above Phase 1's 0.88. *Falsified if* it is ≤ 0.88.
- **H7 — a simplicity prior over dynamics makes lawfulness more typical.** (a) lawful_nt(simplicity, M_A) − lawful_nt(simplicity, Phase 1 rule weights) > 0, for most counted filters, using a paired contrast. (b) lawful_nt(uniform-IC) − lawful_nt(uniform-program) > 0 for most counted filters: the prior over dynamics on its own, with no help from simple initial conditions. This is a risky prediction: M_A's cheapest rules include the chaotic totalistic rules 22, 105, 126 and 150 as well as dead ones.

### 3b.6 Preview, then confirmation
- **Preview (no new simulation).** At r = 1 both machines give equal weight to every member of an ECA symmetry class, so 2a can be computed by reweighting the Phase 1 `full` results. It is labelled a preview because it inherits Phase 1's crossed seeds and the complement approximation (§6).
- **Confirmation run (2a proper):**
  - a new RNG seed, with sampled seeds drawn independently for each rule (so seed clusters nest within rules);
  - the complement approximation removed: embed worlds are run for both rule and complement(rule);
  - long seeds (lengths 4–10, where Phase 1's rich worlds are) oversampled with importance weights, so rich-world intervals are usable;
  - sampled programs checked to be disjoint from Phase 1 `quick` and `full` with `--exclude-programs-from`, with the excluded weight reported;
  - the order is `smoke`, then `quick` (to check compute), then a run the size of `full`.
- **Exit for 2a:** every one of H1–H7 has a verdict under M_A and M_B in both co-primary analyses, plus a table of what changed relative to Phase 1.

### 3b.7 Amendments (pre-registered 2026-11-09, after the 2a preview, before the confirmation run)
The 2a preview (`results/full/report_preview2a.md`) exposed one design flaw, and the review raised several cheap checks. These amendments are **post hoc relative to the preview** (NOTES.md, 2026-11-09). The confirmation run is disjoint from Phase 1 by construction (§3b.6), so it is fresh data for all of them. Where an amendment conflicts with §3b.1–3b.6, the amendment wins. The original text stays as the record.

**A1. Dynamical richness replaces spatial richness as a co-primary analysis.**
- *Why:* the spatial criterion (3-block entropy ≥ 0.5) counts shifts and the identity on a long seed as "rich". Under M_B, 99% of rich-world weight was rules 170, 204, 15 and 51. In the Phase 1 uniform arm, 63% of weight was rich but repeated with translation period ≤ 4.
- *Definition:* a world is **dynamically rich** if its analysed window (steps T0 … T0+T) has no translation period p ≤ 16. That is, no p ≤ 16 and shift |s| ≤ p with x[t+p] = roll(x[t], s) for every t (`worlds.translation_period`). The test excludes fixed points, blinkers, shifts, the identity, and a lone glider moving through empty space. It keeps nested and additive patterns, evolving localised structures, and chaos. Noise worlds always count as rich, as before.
- *How the cutoff was chosen:* from world properties only, with no outcome metric consulted. On the Phase 1 `full` worlds, no translation period falls between 17 and 64; the next group, at 65–256, is the additive rules 105 and 150. So 16 sits in a natural gap.
- *Co-primary analyses are now all worlds and dynamically rich worlds.* The spatial version (§3b.4) is reported as secondary, for continuity with Phase 1.
- *If a prior puts less than 2% of an arm's weight on dynamically rich worlds,* that analysis is reported but flagged as resting on a sliver of the prior. In the preview, M_B puts about 0.3% there. That sliver is itself a result about M_B.

**A2. H6 gets a paired, dynamical version (H6d).**
- *H6 stays as written.* It uses spatial richness and the fixed Phase 1 share of 0.88.
- *H6d:* (share of M_A simplicity weight in non-dynamically-rich worlds) − (the same share under the Phase 1 rule weights, on the same worlds) > 0, using a paired cluster-bootstrap interval. Falsified if the interval lies below 0.
- *Disclosed in advance:* on the preview worlds the point estimate is negative (0.914 vs 0.925). The prediction is registered knowing it currently points the wrong way.

**A3. Radius is priced, not capped: report the truncation.**
- *2a sensitivity M_A\*:* M_A summed over programs at **every** radius that compute the ECA. At r ≥ 2 that means:
  - totalistic tables that are constant (they reach rules 0 and 255);
  - outer-totalistic tables that read only the centre (rules 0, 255, 204, 51);
  - general tables (every ECA, about 2⁻³⁷ each).
  This changes a few class weights by about 6%. It is reported next to M_A as a robustness row and does not get its own verdicts.
- *2b, before it runs:* the prior weight of programs above the largest simulated radius is reported per family. If it exceeds 5% of the machine's total weight, the report says the cutoff could matter, and 2b is not described as covering "the rule space".

**A4. Influence check: leave one rule class out.** For each of the headline verdicts (A6), recompute after dropping each of the 88 rule classes in turn. A verdict that changes when any single class is dropped is reported as **fragile (class k)**. This costs nothing to compute and answers "is this verdict just rules 105 and 150?"

**A5. Fixed-rule intervals alongside.** Every headline contrast also gets a `--fixed-rules` interval: the 88 classes taken as the whole population, not as a sample of possible dynamics. The two intervals answer different questions ("these rules" vs "rules like these"). The superpopulation interval stays the one that decides verdicts (§3b.4).

**A6. Headline vs secondary, to limit the garden of forking paths.**
- *Headline:* H1 and H7a/b under M_A, in the two co-primary analyses (A1), SSA. That is 6 verdicts, each with its A4 fragility label and A5 fixed-rule interval.
- *Secondary, reported in full:* everything else (M_B, M_A\*, SIA, spatial-rich, H2–H6d, all filters).
- *Rule for the write-up:* the 2a summary sentence may only cite headline verdicts.

**A7. Finite-size check.**
- *Which worlds:* the highest-weight dynamically rich worlds in the M_A simplicity arm, taken until they cover 50% of that arm's dynamically rich weight, capped at 40 worlds.
- *What is run:* each is re-run at 2N and 2T, with T0 and the observer grid scaled to match.
- *What is reported:* per world, lawful_nt for each counted filter at both sizes, plus the share of these worlds whose lawful_nt moves by more than 0.1. If that share exceeds one third, every rich-world verdict carries a finite-size caveat.
- *Cost:* about 1–2 CPU-hours.

**A8. Confirmed means confirmed on fresh programs too** (added 2026-11-10, before `quick2a`).
- *Why:* about 79% of `full2a`'s simplicity weight is enumerated short-seed programs that Phase 1 already ran. They are deterministic, so they are the same worlds. A1–A7 were designed after looking at those worlds, and by §5.1 a change that acts mainly through them cannot be confirmed by re-running them.
- *Rule:* a headline verdict (A6) is **confirmed** only if it is SUPPORTED both on all programs and on fresh programs alone. Fresh programs are the simplicity programs that appear in neither Phase 1 `quick` nor Phase 1 `full`, checked in code. The headline table shows both columns.
- *Note:* H7b uses only uniform-arm worlds, which are all new random ICs, so its two columns agree by construction.

---

## 4. Running it

```bash
pip install -r requirements.txt          # numpy, matplotlib (Python 3.10+)
python -m lawlike.selftest                # Phase 0: must say "all checks passed"
python -m lawlike.run --preset smoke      # ~1 min: pipeline works end to end (results meaningless)
python -m lawlike.run --preset quick      # ~10-25 min: first real look
python -m lawlike.run --preset full       # ~1.5-3 h: press go, walk away
python -m lawlike.run --preset deep       # overnight: once full says it's worth it

# Phase 2a confirmation (README 3b.6, 3b.7): smoke2a, then quick2a (compute check), then full2a
python -m lawlike.run --preset full2a     # ~2.5 h on 16 cores; writes report.md and report_2a.md
python -m lawlike.size_check results/full2a   # A7 finite-size check, ~1-2 CPU-hours
```

Times assume a 4–8 core laptop (it uses all cores but one; override with `--workers`). On the dev machine, the cost was about 1.7 s/world for `quick` and 8 s/world for `full`, per core. The Phase 1 `full` run on a 16-thread machine measured 17.6 s/world (21 CPU-hours), so budget for that.

- **Resumable.** Results stream to `results/<preset>/results.jsonl` one world at a time. Ctrl-C at any point, and re-run the same command to continue.
- **Useful while running.** Worlds are processed highest-weight first with the arms interleaved, so `python -m lawlike.analyze results/full` on a half-finished run already gives a fair, noisier estimate.
- **Protected.** A results directory remembers its config hash and refuses to mix runs from different configs.
- **Microscope.** To see what a filter is actually picking:
  ```bash
  python -m lawlike.inspect_world --rule 110 --kind uniform --filter individuality --out r110.png
  python -m lawlike.inspect_world --rule 54 --kind embed --seed 1 --filter integration_excl
  python -m lawlike.inspect_world --kind noise --filter causal_complexity
  ```
  This shows the spacetime, the causal-complexity field, and the admitted observers' tracks coloured lawful, noisy, or trivial.

### Outputs (in `results/<preset>/`)
- `report.md`: tables, a verdict per hypothesis, filter health, and which rules host the lawful observers.
- `summary.csv`: every (weighting, filter, arm, metric) with a 95% cluster-bootstrap interval.
- `report_disjoint-<other>.md`, `summary_disjoint-<other>.csv`: written by `python -m lawlike.analyze results/<preset> --exclude-programs-from results/<other>`, the same analysis restricted to world programs that don't appear in the other run (§5.1).
- `plots/`: lawfulness by filter, filter agreement heatmap, and which observer families each filter admits.
- `observers/*.npz`: the full per-observer table for every world (scores, metrics, admission masks), so new analyses don't need a re-run.
- `config.json`, `errors.log`.

`example_results/quick/` holds the report and plots from one `quick` run on the development machine, so you can see what a finished run looks like before starting your own. Your numbers should match closely, since everything is seeded.

---

## 5. Guarding against the wrong mountain

You can't remove bias; the realistic goal is to make every choice visible, and avoid quietly fitting the method to the hoped-for answer.

1. **Pre-register.** Hypotheses and thresholds live in this README and in code *before* a run. If you change one after seeing results, log it in `NOTES.md` with the reason, and then test it on **fresh data**: world programs disjoint from the run that suggested the change. Check this in code, not by eye: `python -m lawlike.analyze <new> --exclude-programs-from <old>`, and report the excluded weight. *A bigger preset is not automatically fresh data.* Under a simplicity prior most of the weight sits in the few shortest programs, and every preset enumerates them (81% of `full`'s simplicity weight was the same programs as `quick`). Those enumerated programs are the population, not a sample. Re-running them at larger N and T tests robustness to size, not replication. So a post-hoc change that matters only through them can't be confirmed by any fresh run, and the log should say so.
2. **Controls are first-class.** The null observer family, the uniform arm, and the noise arm are in every run. A criterion that doesn't separate structured worlds from noise, or that admits mostly null or frozen observers, is not measuring observerhood, however elegant it is.
3. **Prune on stated reasons, never on disagreement alone.** The filter-health table checks (a) world-ignoring observers excluded, (b) not mostly frozen streams, (c) structured vs noise separated. Agreement with other filters is reported but is **not** a pruning criterion by itself. Pruning the odd one out would manufacture H3.
4. **Report every filter, every run.** Dropped filters stay in the code and in the report, marked as deprecated rather than deleted, so a later substrate can resurrect them.
5. **Both anthropic weightings, always.** A conclusion that holds only under one weighting gets stated that way.
6. **Look before you believe.** When a table says a filter loves some rule, run `inspect_world` on it. Several design fixes in this codebase came from looking: the causal-state merge threshold was too strict for small samples, and observers needed fractional speeds to ride gliders.

---

## 6. Known limitations (Phase 1)

- **Finite everything.** T ≤ 2048 steps, N ≤ 256 cells, Markov order ≤ 8, light cones of depth 3/2. The theorems are asymptotic; this is a finite-sample shadow of them.
- **Estimator bias.** Plug-in entropies with Miller–Madow correction and stratified-permutation debiasing are standard, but large alphabets (causal-state observers) at short T are still under-sampled. Treat small score differences as noise.
- **The integration measure is a proxy**, not IIT phi. The blanket measure tests a single horizon (2 steps).
- **The coder is a mixture of Markov models**, not a universal prior. It can't see long-range or non-local structure, so some lawful streams (for example, rule 90's nested patterns) will look noisier than they are. Context-tree weighting or a small neural predictor is a natural upgrade.
- **The causal-state cap** (48 states) is hit by some rules, such as rule 110's ether. Raise it in `config.py` if inspection shows merged structures.
- **Small rings die.** Rule 18 and others can annihilate all defects on a small periodic lattice, leaving observers watching a world that has ended. That's real dynamics, not a bug, and loosely the "dissolution" scenario from the discussion.
- **Every rule costs 8 bits**, so Phase 1 can't test whether simpler *dynamics* are favoured. That's Phase 2.
- **Complement approximation.** Each symmetry class is simulated once (its representative) and weighted by class size. That's exact for reflection and for uniform initial conditions. It isn't exact for `embed` seeds under complement: complement(rule) run on a seed embedded in zeros is equivalent to rule run on a seed embedded in *ones*, which is not in the IC set. Found in review on 2026-10-23; removed in the Phase 2a confirmation run (§3b.6).
- **Single-seed strata.** The `quick` preset samples one seed per long-seed length, so the bootstrap cannot see that part of the sampling variance (the report notes this).

---

## 7. Code map

```
lawlike/
  config.py         presets (smoke / quick / full / deep) and every numeric convention
  worlds.py         ECA rules & symmetry classes, simulation, the three world priors
  causal.py         local causal states and local statistical complexity
  observers.py      observer families, description lengths, stream extraction
  info.py           entropy / MI / CMI estimators, integration proxy, KT-mixture coder, NCD
  experiment1.py    per-world pipeline: filters, law metrics, per-filter summaries
  run.py            parallel, resumable runner
  analyze.py        aggregation, hypothesis verdicts, plots; --exclude-programs-from for disjointness
  bootstrap.py      cluster bootstrap matched to the sampling design (rules x sampled seeds)
  rule_priors.py    Phase 2 rule priors: phase1 (uniform), M_A (family table), M_B (min Boolean formula)
  preview2a.py      Phase 2a preview: Phase 1 results reweighted under M_A / M_B (§3b.6)
  analyze2a.py      Phase 2a analysis: priors x analyses, headline, fragility, fixed-rule intervals (§3b.7)
  size_check.py     Phase 2a finite-size check (§3b.7 A7)
  inspect_world.py  the microscope
  selftest.py       Phase 0 checks
```

Adding a filter means writing a scoring function in `experiment1.observer_scores`, adding its name to `SCORE_FILTERS`, and adding a sanity case to `selftest.py`. Adding a substrate (Phase 2) means a new `worlds_*.py` that returns spacetime arrays. Observers, filters and metrics carry over as long as the world is a **binary, radius-1** 1-D lattice over time. Light-cone shape (`causal._cone_keys`) and the width of the observer boundary and environment (`observers.Extractor.read`) assume radius 1, and coarse-graining, the boundary encoding and causal-cone keys assume two colours. Phases 2b and 2c generalise these first (§3b.1).

---

## 8. References

- M. P. Müller, "Law without law: from observer states to physics via algorithmic information theory," *Quantum* 4, 301 (2020); arXiv:1712.01826.
- R. J. Solomonoff, "A formal theory of inductive inference" (1964); "Complexity-based induction systems" (1978).
- J. Schmidhuber, "The Speed Prior: a new simplicity measure yielding near-optimal computable predictions" (COLT 2002).
- A. Rupe & J. P. Crutchfield, "Local causal states and discrete coherent structures," *Chaos* 28, 075312 (2018).
- C. R. Shalizi et al., "Automatic filters for the detection of coherent structure in spatiotemporal systems," *Phys. Rev. E* (2006).
- D. Krakauer, N. Bertschinger, E. Olbrich, J. Flack, N. Ay, "The information theory of individuality," *Theory in Biosciences* (2020).
- K. Friston, "Life as we know it," *J. R. Soc. Interface* (2013).
- M. Biehl, F. Pollock, R. Kanai, "A technical critique of some parts of the free energy principle," *Entropy* (2021).
- N. Ay, "Information geometry on complexity and stochastic interaction," *Entropy* (2015).
- M. Oizumi, L. Albantakis, G. Tononi, "From the phenomenology to the mechanisms of consciousness: IIT 3.0," *PLoS Comput. Biol.* (2014).
- R. Krichevsky & V. Trofimov, "The performance of universal encoding," *IEEE Trans. Inf. Theory* (1981).
- R. Cilibrasi & P. Vitányi, "Clustering by compression," *IEEE Trans. Inf. Theory* (2005).
- S. Wolfram, "A class of models with the potential to represent fundamental physics," *Complex Systems* 29 (2020).
- S. M. Carroll, "Why Boltzmann brains are bad," arXiv:1702.00850 (2017).
- G. Egan, *Permutation City* (1994) — dust theory, dramatised.
