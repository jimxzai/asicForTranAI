# asicForTranAI — Strategy & Business Plan Update (Q4 2026)

**Date**: 2026-10-22 · **Scope**: the verified low-bit inference stack only (Fortran/MLIR kernels, SPARK, Lean 4, `2025-3.5bit-groq-mvp/`).
Gazillioner is a separate repo ([svden3/gazillioner](https://github.com/svden3/gazillioner)) with its own plan; a Q4 2026 review of it is at `archive/local-first-device/docs/STRATEGY_2026_Q4.md` there.

**Supersedes**: `STRATEGIC_UPDATE_DEC2025.md`, `STRATEGY_2025_UPDATE.md`, and the 2026–2027 sections of `VISION_2025_2032.md` / `BRD_Business_Requirements.md`. Those stay for history.

Facts below marked † come from web searches on 2026-10-22 and were mostly secondary sources; re-check anything you will quote externally.

---

## 1. What changed since the Dec 2025 plan

| Development | Implication for this project |
|---|---|
| Nvidia licensed Groq's LPU IP and hired ~80% of its staff (~$20B); Groq 3 LPU ships inside Vera Rubin from Q3 2026; Groq's remaining valuation reportedly ~$3.5B, with a shareholder suit and a reported DOJ inquiry † ([Tom's Hardware](https://www.tomshardware.com/tech-industry/semiconductors/nvidias-20-billion-groq-deal-produces-its-first-chip), [lawsuit](https://www.tomshardware.com/tech-industry/semiconductors/former-groq-engineers-sue-board-over-usd20-billion-nvidia-deal-saying-it-handed-nvidia-the-lpu-and-the-team-that-built-it-plaintiffs-allege-the-board-kept-billions-from-other-shareholders)) | "Groq partnership" is no longer a strategy. Treat the remaining Groq as a cloud API, not a partner. `GROQ_EMAIL_DRAFT.md` is obsolete. |
| Cerebras IPO'd (May 2026, ~$5.55B raised) † ([Forklog](https://forklog.com/en/cerebras-systems-secures-5-55-billion-in-2026s-largest-ipo/)) | A funded public company with a software ecosystem to fill. A realistic design-partner target. |
| AMD agreed to acquire Taalas (Aug 2026), whose HC1 hardwires Llama 3.1 8B into silicon (~17k tok/s, company claim) † ([NYU summary](https://rits.shanghai.nyu.edu/ai/amd-acquires-taalas-the-startup-etching-llms-into-silicon/)); Etched raised ~$500M (Jan 2026) † | Inference ASICs are consolidating into large vendors. Model-specific silicon makes *numerical correctness of fixed weights* a one-shot, high-stakes problem, which is a verification opportunity. |
| FP4 is mainstream: NVFP4 (4.5 effective bits) and MXFP4 (4.25) are native on Blackwell/MI355X, and vendors now ship QAT checkpoints in these formats † ([field guide](https://tensorfoundry.io/blog/llm-quantisation-field-guide)) | A post-training "3.5-bit" format is not a moat. Re-quantizing vendor QAT checkpoints also degrades them (one measurement: ~3× KL divergence) †. Compete on *proofs about* these formats, not on bit-width alone. |
| Formal-verification-for-AI money arrived: Axiom $200M Series A at $1.6B (Mar 2026), Harmonic (>$200M raised), Pramaana $27M seed for a verification layer targeting tax/legal/healthcare † ([TAMradar](https://www.tamradar.com/funding-rounds/axiom-series-a-200m), [Pramaana](https://www.tamradar.com/funding-rounds/pramaana-labs-seed-27m)) | The Dec 2025 claim of "ZERO competitors" is stale. Those firms verify *reasoning, math and generated code*; none I found verify *low-precision numerical kernels*. That niche is open, but the category is now funded and visible. |
| SBIR/STTR lapsed 6 months and was reauthorized to 2031 on 2026-04-13; adds Strategic Breakthrough awards (up to $30M), per-agency proposal caps from FY2027, and tougher foreign-affiliation screening † ([Mondaq](https://www.mondaq.com/unitedstates/government-contracts-procurement-ppp/1774902/sbirsttr-programs-reauthorized-after-six-month-lapse)) | The "apply to NSF SBIR in Jan 2026" step could not have happened. Programs are open again now, and caps make early, well-targeted submissions valuable. |
| EU AI Act omnibus (provisional): Annex III high-risk to 2027-12-02, Annex I to 2028-08-02; formal adoption status unconfirmed † ([Gibson Dunn](https://www.gibsondunn.com/eu-ai-act-omnibus-agreement)). EUROCAE ED-324 / SAE ARP6983 (airborne ML assurance) planned for ~Q4 2026 † | Compliance-driven demand moves out ~1 year. Aviation certification is a 2028+ story; track ED-324 instead of promising DO-178C. |
| DRAM/NAND prices up as much as ~89% QoQ in Q2 2026; tightness expected through 2027–28 † ([TweakTown](https://www.tweaktown.com/news/112358/dram-prices-surged-by-up-to-89-in-q2-2026-destroying-the-consumer-segment/index.html)) | Memory is the scarce input, so "fewer bits per weight" is worth more than in 2025. This helps the core value proposition but not hardware-selling plans. |
| Q3 2026 venture: $159B globally, 64% to AI, record 27 $1B+ rounds † ([summary](https://fourweekmba.com/ai-crunchbase-q3-2026-159b-venture-27-billion-dollar-rounds-ai/)) | Capital is abundant but concentrated in mega-rounds. A solo-founder tooling company should plan for grants + services + a small seed, not a large round. |

## 2. Thesis reset

**Old**: "World's first 3.5-bit Fortran stack on Groq, formally verified, 12–24-month lead."
**New**: **"Machine-checked guarantees for low-precision inference: proofs that quantized kernels (MXFP4, NVFP4, INT4, 3.5-bit) are overflow-free and within stated error bounds, delivered as an open core plus an evidence package for chip vendors and regulated deployers."**

Rules that follow:
1. **Bit-width becomes a parameter, not the brand.** Extend the Lean/SPARK work to block-scaled FP4 formats, because that is where the market is. Keep 3.5-bit as one verified instance and as a research result.
2. **MXFP4 first.** `studies/format_error_study.py` shows the 3.5-bit pair format loses to INT4 at equal bits, while block-scaled FP4 wins on heavy tails; the verified-kernel work targets MXFP4/NVFP4, with the 3.5-bit format kept as a research variant. See [EVALUATION_2026-10.md](EVALUATION_2026-10.md).
3. **Hardware-agnostic.** No dependency on any one vendor. Targets: Cerebras, AMD (post-Taalas), Tenstorrent, Hailo, Apple/Nvidia edge parts.
4. **Fortran/MLIR is an implementation detail**, not a selling point. Lead with proofs and artifacts.
5. **Claim only what is machine-checked.** "Overflow-free" and "error ≤ ε for these formats on these ops" are provable. "No hallucinations" or "correct advice" are not. This also protects the Gazillioner side from AI-washing exposure.

## 3. Business model (sequenced)

| Stage | Offer | Buyer | Price hypothesis (unvalidated) | Why now |
|---|---|---|---|---|
| A. 0–6 mo | Funded R&D: SBIR/STTR Phase I (AFWERX, DARPA, NSF) on "verified low-precision inference" | US gov | Phase I typically ~$250–300K; check current solicitations | Programs reopened Apr 2026 |
| A. 0–6 mo | Kernel verification audit (one op or one transformer layer, fixed format) with a Lean + SPARK report | Chip/IP vendors, edge-model startups | $40–150K per engagement | Taalas-style fixed-weight silicon needs pre-tapeout assurance |
| B. 6–18 mo | Licensed proof library + CI checker ("verified-kernels") with a free open-source tier | Chip vendors, automotive/medical edge teams | Annual license, tiered by number of targets | Converts audits into recurring revenue |
| C. 18 mo+ | Certification evidence packages aligned to ED-324 / ISO 26262 / EU AI Act | Avionics, automotive Tier-1s | Project-based, six figures and up | Standards land 2026–2028 |

Services pay for the work; the proof library is the asset. Do not hire before a second paying customer.

## 4. Keep / change / stop

- **Keep**: Lean 4 quantization-bound proofs, SPARK overflow proofs, open-source core, the credibility story (1990 Fortran award, SGI, Peter Chen).
- **Change**: lead messaging from "3.5-bit" to "verified FP4/INT4 inference"; benchmark against MXFP4/NVFP4 rather than AWQ-INT4; replace the 405B/1T-parameter ambitions with small-model scope (8B-class), which is where fixed-weight silicon and edge devices are.
- **Stop**: Groq-specific outreach; "zero competitors" language; the 2029–2032 consulting and retirement timeline as a planning input (keep it as a personal vision doc only); smart-glasses and robotics side-tracks; any marketing claim not backed by a reproducible artifact.

## 5. 90-day plan (Nov 2026 – Jan 2027)

| Window | Deliverable | Done when |
|---|---|---|
| Weeks 1–2 | **Claims audit**: DONE in first pass, see [CLAIMS_AUDIT_2026-10.md](CLAIMS_AUDIT_2026-10.md) (7 failures: 19 GB is arithmetically wrong, 4,188 tok/s is an estimate, "247 proofs" is hard-coded output, 37 Lean `sorry`). Done since: `kernel3p5/` reference kernel (exhaustive tests, measured 3.5 bits/weight, CPU benchmark), README and public pages corrected, CI made real. Remaining: install GNATprove and Lean, rerun, publish real proof counts | `scripts/claims_audit.sh` passes or its failures are explained; public pages corrected |
| Weeks 1–3 | Extend Lean model to MXFP4/NVFP4 block-scaled dot products | Theorem statements and at least one fully proved bound |
| Weeks 2–6 | Two SBIR/STTR topic matches identified; one white paper or Phase I submitted | Submission confirmation |
| Weeks 3–8 | 20 targeted outreaches to chip/IP vendors (Cerebras, AMD/Taalas, Tenstorrent, Hailo, Axelera, Etched) with a 2-page "verified FP4 kernel" brief | ≥5 conversations, ≥1 paid pilot scoped |
| Weeks 6–12 | Preprint on verified block-scaled quantization (confirm next conference deadlines; NeurIPS 2026 has passed) | arXiv posted |

**Gates**: no paid pilot or grant submission by end of Dec → cut scope to research-only and re-plan runway. Pilot signed → hold everything else constant and deliver it.

## 6. Risks

| Risk | Mitigation |
|---|---|
| Nvidia/AMD/Cerebras build verification in-house | Open-source the checker; sell cross-vendor neutrality and independence |
| Proofs cover idealized arithmetic, not real hardware | State assumptions explicitly; verify at the instruction/IR level where possible |
| Funded Lean companies pivot into numerics | Move fast on the niche; partner with or cite their tools instead of competing on general provers |
| SBIR foreign-affiliation screening and proposal caps | Prepare the ownership/affiliation disclosures now |
| Solo-founder bandwidth, split with Gazillioner | Time-box: this project gets the technical deep work; Gazillioner gets scoped product sprints (see its plan) |

## 7. Open questions for you

1. What is the runway in months at current burn? It decides whether the grant path (slow) or the services path (faster) leads.
2. Are the 2025 benchmark numbers still reproducible on your hardware?
3. Was the NeurIPS 2026 draft submitted? If not, which venue is next?
4. Any existing warm contacts at Cerebras/AMD/Tenstorrent?

## Sources
Searches run 2026-10-22; see inline links. Unverified or thin areas: Etched figures (aggregator only), Harmonic's totals (sources disagree), Ministral 3 adoption (no results), ED-324 release date (working-group deck says Q4 2026; another source says June 2026).
