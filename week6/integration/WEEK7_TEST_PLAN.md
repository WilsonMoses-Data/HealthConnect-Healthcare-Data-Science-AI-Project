# Week 7 testing requirements

Freeze healthconnect-week6-v1 and retain the Week 5 benchmark before collecting or evaluating new data. Do not treat the reused 799 records as fresh evidence.

| Area | Test or evidence | Acceptance condition | Status |
|---|---|---|---|
| Fresh temporal evaluation | Later labelled appointments with known outcome windows | Same eligible population and locked candidates; report ranking, calibration and threshold trade-offs | New data required |
| Schema | Seven fields, explicit dates and documented missingness | Valid batches preserve IDs; invalid batches fail clearly | 17 local checks passed; realistic batches pending |
| Model parity | Reload artifact and compare single/batch scores | Agreement within 1e-12 in the pinned environment | Passed locally |
| Source integrity | Compare original hash and feature provenance | Original data unchanged; all scoring fields known at booking | Source preserved; operational provenance pending |
| Patient history | Confirm stable identity and repeated-patient structure | Define valid uncertainty/group evaluation methods | Source clarification required |
| Cancellations | Define eligible appointments at scoring time | Document how cancellations affect training, labels and evaluation | Unresolved |
| Sunday scheduling | Confirm clinic hours and recorded appointments | Confirm correction or justified inclusion policy | Unresolved |
| Missing and unseen inputs | Distance missing, new appointment category and out-of-range age | Documented behaviour; no silent failure | Basic checks passed |
| Fairness | Counts, recall, false-positive rate and calibration across groups | Stakeholder-approved criteria and sufficient support | Criteria not yet supplied |
| Workload | Threshold and queue volume on later data | Agreed staff capacity and escalation procedure | 30% scenario only |
| Reliability | Repeated batches, failure logs, latency and environment recreation | Technical performance targets agreed before testing | Targets pending |
| Intervention | Supportive follow-up and outcome recording | Staff-approved pilot; evaluate attendance impact separately | Not started |
| Collaboration | Review with programme or real track participant if required | Confirm acceptance of simulated integration or record actual exchange | Pending programme confirmation |

Do not invent fixed business success thresholds. Define them with the clinic/mentor before judging operational suitability. A technical pass permits further testing, not autonomous clinical use.
