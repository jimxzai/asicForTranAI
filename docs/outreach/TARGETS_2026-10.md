# Target list for the 14-day test (organizations and roles only)

Built 2026-10-22 from web searches; secondary sources, so re-check anything you quote. **No individuals are named**: find the right person through your network, the team's public papers or repos, or conference programs. "Fit" is my guess, to be replaced by what the calls show. Log results in `LOG.md`.

Evidence used: Hailo-10H supports INT4 and is AEC-Q100 Grade 2 qualified ([edge chip survey](https://aimultiple.com/edge-ai-chips)); NVIDIA DRIVE Thor lists FP4/INT4 with ASIL-D use cases ([Edge AI and Vision Alliance](https://www.edge-ai-vision.com/2025/09/accelerate-autonomous-vehicle-development-with-the-nvidia-drive-agx-thor-developer-kit/)); AMD agreed in Aug 2026 to acquire Taalas, closing Q4 2026 pending approval ([NYU summary](https://rits.shanghai.nyu.edu/ai/amd-acquires-taalas-the-startup-etching-llms-into-silicon/)); Cerebras IPO'd May 2026 ([Forklog](https://forklog.com/en/cerebras-systems-secures-5-55-billion-in-2026s-largest-ipo/)); AdaCore partners with NVIDIA on Ada/SPARK for safety-critical firmware and published AI-code guardrail work in 2026 ([AdaCore](https://www.adacore.com/company/partners/nvidia), [UST](https://www.unmannedsystemstechnology.com/2026/09/adacore-demonstrates-deterministic-guardrails-for-ai-high-integrity-software-development/)).

## A. Chip and IP teams (primary)

| # | Organization | Why them | Role to approach | Fit |
|---|---|---|---|---|
| 1 | Hailo | INT4 inference parts; automotive qualification under way | Compiler / quantization lead; functional-safety lead | High |
| 2 | Axelera AI | Edge inference accelerator; software stack needs assurance story | Quantization / SDK lead | Med |
| 3 | SiMa.ai | $150M Series C (reported); edge AI for robots, drones, cameras | Software/ML compiler lead | Med |
| 4 | AMD (Taalas integration) | Fixed-weight silicon; pre-tape-out correctness is high stakes; deal closing Q4 2026 | Architect on the Taalas/AI-inference side (approach after close) | High, timing-dependent |
| 5 | Etched | Transformer-specific ASIC; large raise reported Jan 2026 (aggregator) | Architecture / numerics lead | Med |
| 6 | Tenstorrent | Open RISC-V AI stack; verification-minded culture | Software/numerics lead | Med |
| 7 | Cerebras | Public and well funded, but a poor first customer: weights live in huge on-chip memory so compression matters less; FP8/FP4 inference support unconfirmed; no safety-critical demand found; business is hyperscale cloud and a few large customers. Value is credibility and learning. SDK and CSL are public with a simulator, so a bit-exact conformance demo is cheap. Contact developer relations only after that demo exists. | Developer relations; numerics / compiler team | Low-Med |
| 8 | NXP (i.MX NPU / eIQ) | Embedded NPU in automotive and industrial parts | eIQ / ML software product owner | Med (check safety scope) |
| 9 | Ambarella | AI vision SoCs for automotive and cameras | ML toolchain lead | Low-Med (check scope) |
| 10 | NVIDIA (DRIVE / Jetson safety teams) | FP4/INT4 with ASIL-D claims; large and in-house, so low odds, but a reference customer if it lands | Functional-safety software lead | Low |

## B. Safety-assurance buyers (second)

| # | Who | Why | How to find the person | Fit |
|---|---|---|---|---|
| 11 | Members of the EUROCAE / SAE joint committee behind ED-324 / ARP6983 (airborne ML assurance) | They are writing the standard and know which arithmetic evidence is asked for. Verify the group name and member list on the EUROCAE/SAE sites. | Published member lists, committee papers, conference talks | High for learning, Med for revenue |
| 12-14 | Three named automotive Tier-1 ML-perception safety teams (pick from conference speaker lists) | ISO 26262 reviewers meet ML kernels | ADAS / perception safety engineer or assessor | Med |
| 15 | One medical-device AI team under IEC 62304 | Class C software needs rigorous evidence | Software-safety lead | Low-Med |

## C. Ecosystem and channel

| # | Organization | Why | Role | Fit |
|---|---|---|---|---|
| 16 | AdaCore | Makes SPARK/GNATprove; published 2026 work on verifying AI-generated code; NVIDIA safety-firmware partner. Could be a channel, a co-author, or a source of a qualified-tool path. | Product manager for SPARK and AI | High as partner, not a payer |
| 17 | Lean community / Lean FRO contacts | Reviewers for the Lean proofs and `bv_decide` trust note; credibility | Maintainers / community leads | Credibility only |

## D. Funders

| # | Program | Why | Action | Fit |
|---|---|---|---|---|
| 18 | AFWERX SBIR/STTR | Reauthorized April 2026; avionics assurance angle | Read open topics; email the topic point of contact with the fit question | Med |
| 19 | DARPA (formal-methods or assured-autonomy program managers) | Verification of ML components | Find current program managers on DARPA pages | Med |
| 20 | NSF SBIR | Research-grade topic: verified low-precision inference | Check current solicitation and email the program officer | Med |

## Notes
- **Sequencing:** start with #1, #3, #6, #16, #18, #11 (edge-chip vendors first because they are memory-bound and ship INT4; Cerebras later, after a conformance demo): they are the most likely to answer and the most informative. Hold #4 until the AMD–Taalas close.
- **Not verified:** ISO 26262 ratings and FP4/INT4 support for Axelera, SiMa.ai, NXP, Ambarella and Mobileye; the exact name and membership of the ED-324 committee; current DARPA program names. Check before the first email.
- **Not on the list on purpose:** verification startups (Axiom, Harmonic and similar) are peers, not customers; Groq (remaining entity after the Nvidia license deal) is a cloud API, not a partner.
- **Tone:** lead with the one-pager and reproducible commands, state the limits first, ask for 20 minutes.
