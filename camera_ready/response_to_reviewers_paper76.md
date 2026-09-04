# Internal response to reviewers — Paper 76

The manuscript now opens with a concrete panel-expansion scenario, explicitly distinguishes unseen-episode deployment from within-episode forecasting, adds a roadmap and focused related work, and ends with conclusions comparing its contribution with leakage and structured-CV literature.

The central result was checked in `results/protocol_main_30/main_benchmark.csv` (30 paired seeds): row-wise AUROC 0.835 versus grouped 0.552, paired gap 0.283 [0.246, 0.320]. The stronger-signal check is in `results/protocol_main_30/highsignal_benchmark_latest.csv` (30 paired seeds): grouped 0.723, row-wise 0.922, gap 0.199 [0.170, 0.229]. The robustness grid contains 10 replicates in every one of 81 cells, and the drift-DGP normalization experiment contains 30 replicates per model/condition cell. The paper reports the temporal split result (0.917) and explains that it represents known-episode forecasting, not unseen-episode generalization.

The corrected mechanism-isolation table uses a true matched-prevalence negative control: fingerprint-off removes static episode features, and event-time-off uses fixed-length episodes with independent row outcomes. The off/off gap is -0.001 [-0.008, 0.006]; the joint on/on condition is 0.283 [0.246, 0.320]. The paper reports this interaction without attributing the headline gap to either component alone.

The abstract and limitations now explicitly describe the study as simulation-based. The formal theorem is a remark. The outstanding item is production readiness, not an unaddressed scientific response: authoritative author metadata and a compliant page count are required before upload.
