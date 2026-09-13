# Understanding the HealthConnect journey from Week 4 to Week 6

This guide explains the reasoning behind the work. Keep the three notebooks open alongside it. Read one section, answer its checkpoint, then move on. The Week 6 notebook includes additional prompts beside each code block.

## 1 Start with the decision

In Week 4, the clinic problem was missed appointments and inefficient use of appointment slots. The Data Science question was whether information available before an appointment could identify no-show risk. The proposed action was supportive follow-up. These are three different things: a business problem, an analytical question and an intervention.

IBM stage: Business Understanding and Analytical Approach. Python was not yet the main decision. A perfectly coded model would still be unhelpful if nobody could act on its output.

Checkpoint: If a model correctly flags an appointment but no staff member follows up, what evidence do we have that the clinic benefited?

## 2 Decide what a row and a target mean

One row is an appointment. A patient can have several appointments, which affects how independent the observations are. The initial binary target is No-Show = 1 and Attended = 0. Cancellations remain in the original data but are excluded from this experiment.

The important limitation is timing: at booking, we cannot know which appointment will later be cancelled. A model trained only on eventual attendance/no-show outcomes does not automatically describe every future booking.

IBM stage: Data Requirements and Data Understanding. A target is a business definition expressed as code, not merely a convenient column to predict.

Exercise: Explain why changing Cancelled to 0 would redefine the problem rather than simply clean the data.

## 3 Read data without changing its meaning

Week 4 found that the literal reminder value None means no reminder channel. Pandas can interpret certain strings as missing by default. Reading with keep_default_na=False protects that category; numerical blanks are then handled explicitly.

Dates also need an explicit month/day/year format. Sunday appointments conflict with the clinic description, and repeated patient IDs have inconsistent ages. We preserved the records and documented those conflicts rather than inventing corrections.

IBM stage: Data Collection and Data Understanding. The dataset, dictionary and operational description must be checked against each other.

Checkpoint: What is the difference between a value being unusual and being demonstrably wrong?

## 4 Make the experiment resemble the prediction moment

Week 5 refined prediction to booking time. Its training appointments ended before 1 March 2026. Test appointments were booked from that date onwards. Appointments booked before the boundary but occurring after it were purged because their outcomes would not have been known at the boundary.

There were 3,682 training records, 799 comparison records and 256 purged records. The preparation pipeline fitted medians, scaling and category encoding on training records only.

IBM stage: Data Preparation and Evaluation design. The sequence is: define what would be known, define partitions, then fit learned transformations. Fitting a median on all data would let future records influence training.

Exercise: An appointment is booked on 25 February and occurs on 5 March. Why is its attendance outcome unsuitable for training a model at the 1 March boundary?

## 5 Establish a baseline that can be challenged

Week 5 compared logistic regression with an all-no-show majority dummy. The dummy caught every no-show by flagging every appointment. Its recall and F1 were higher, but its workload was also much larger. Logistic regression improved accuracy and ranking while making both false-positive and false-negative errors.

Precision asks: of the appointments flagged, how many actually became no-shows? Recall asks: of all actual no-shows, how many were flagged? ROC-AUC describes ranking across thresholds. None of these metrics alone measures an attendance improvement caused by staff support.

IBM stage: Modelling and Evaluation. Baselines prevent sophisticated code from being mistaken for useful progress.

Checkpoint: Why does perfect recall from the dummy not make it the obvious operational winner?

## 6 Week 6 tests ideas without choosing on the old test set

We had already examined the Week 5 comparison set. It cannot become a fresh holdout again. Week 6 therefore used three expanding temporal folds within the original training data, comparing the same baseline, no-history logistic regression, random forest and gradient boosting.

Every fold fitted its own preprocessing. Candidate selection used mean ROC-AUC; mean Brier score was the tie-breaker. The no-history model ranked first at 0.6481, compared with 0.6416 for the baseline. This small difference is not proof of statistically significant superiority.

IBM stage: Modelling and Evaluation. Predeclaring the selection rule makes it harder to choose whichever result happens to look attractive afterwards. The selected model's internal score is still a selection score, so an independent later evaluation remains necessary.

Exercise: Random forest has higher accuracy on the reused comparison set. Why did we not switch to it after looking at that result?

## 7 Simpler features can be a meaningful improvement

The chosen model removed previous appointment count, prior no-show rate and the no-history flag. Their timing had not been verified. It retained six inputs to the fitted preprocessing: age, distance, weekend status, log lead days, appointment type and appointment time.

This reduces a dependency on uncertain history and slightly improves internal ranking. On the reused comparison, however, AUC is slightly lower than Week 5. A responsible interpretation reports both findings.

IBM stage: Data Preparation revisited. Methodology is iterative. New evidence can take us back to feature requirements and then forward into modelling again.

Checkpoint: Name one implementation benefit and one performance limitation of choosing the no-history model.

## 8 Errors reveal trade-offs rather than patient motives

At threshold 0.5 the candidate catches 260 no-shows and misses 127, compared with 246 caught and 141 missed by Week 5. It also produces 184 false positives instead of 172. That is 14 more no-shows identified in exchange for 12 more false positives.

False negatives have median lead time 12 days; false positives 34 days. These patterns suggest questions for further analysis, not reasons for any individual's behaviour. Segment rates need their denominators and sample sizes. Removing gender from inputs does not establish fairness.

IBM stage: Evaluation and renewed Data Understanding.

Exercise: Write a two-sentence explanation of these results to a receptionist without using the words AUC, classifier or hyperparameter.

## 9 A probability and an action rule are separate

The model returns a probability estimate. A threshold converts that estimate into a flag. Week 6 explored a hypothetical 30% validation-workload limit and chose 0.600. On the reused comparison this flags 193 appointments, with precision 70.47% and recall 35.14%.

This reduces follow-up volume but misses 251 no-shows. The scenario is not clinic-approved and does not guarantee a future workload percentage. A top-k queue would be a different action policy and would require its own validation and capacity rules.

IBM stage: Evaluation linked back to Business Understanding.

Checkpoint: If the clinic wants fewer follow-ups, what loss must it consider alongside the higher precision?

## 10 Integration makes assumptions executable

The AI-simulated ML Engineering review asked whether another component could score raw appointment records reliably. The result is a score function and command-line interface, a saved preprocessing/model artifact, a seven-field raw schema, sample input/output and 17 tests.

The raw schema has more fields than the fitted predictor list because appointment ID is returned for traceability and two dates are used to derive features. Those three raw columns are not directly passed as predictors. Histories and outcomes cannot influence this candidate's scores.

IBM stage: Deployment preparation. A local scoring interface is not a deployed service. Serialization is not proof of operational readiness. In this task, the counterpart was ChatGPT in a simulated role, not a human intern; programme acceptance must be confirmed.

Exercise: Delete the age column from a copy of integration/sample_input.csv and run the scorer. Read the failure message. Then restore it and make only distance blank. Explain why the two cases behave differently.

## 11 What Week 7 must establish

We now have a candidate, not a finished clinic solution. Week 7 needs fresh evaluation records, contract checks on realistic data, subgroup/calibration review, capacity agreement, a cancellation policy and a controlled supportive workflow. Results must remain versioned and traceable.

The reusable reasoning pattern is: define the decision, establish what is knowable, protect evaluation data, compare alternatives fairly, interpret trade-offs, validate the interface, and state what evidence is still missing.

Start our guided discussion with section 1. We should pause at each checkpoint rather than rush through every notebook at once.
