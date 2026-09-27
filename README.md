# HealthConnect Week 8 Final Analytics

**Wilson Moses | AnalystLab Africa Experience Lab | Week 8**

This package presents a final descriptive analytics contribution to the HealthConnect Clinic Experience Lab. It supports the question of how HealthConnect could use data and AI to reduce missed appointments and improve patient support.

## Main finding

The supplied fictional dataset contains 5,000 appointment records and 1,696 distinct anonymized patients. Among 4,737 completed appointments, 2,423 were recorded as no-shows (51.2%). The no-show rate rises from 26.6% for bookings 0-3 days ahead to 63.9% for bookings 31 or more days ahead. This is a descriptive pattern in synthetic data and does not establish causation or real clinic performance.

## Track and continuity note

The supplied Week 8 assignment is for Data Analytics. Available Week 4-6 project records describe Wilson Moses's Data Science work. This package treats the analytics output as a cross-track contribution. No Week 7 Data Analytics report or test result was present in the reviewed materials, so this package does not claim a Week 7 test or refinement.

## Package contents

- report/HealthConnect_Week8_Final_Analytics_and_Decision_Support_Report.pdf
- report/HealthConnect_Week8_Project_Summary.pdf
- presentation/HealthConnect_Week8_Final_Presentation_and_Learning_Guide.pptx
- analytics/HealthConnect_Week8_Analytics_Notebook.ipynb
- analytics/src/analyze_healthconnect.py
- analytics/outputs/ KPI, segment and source-quality tables plus figures
- dashboard/HealthConnect_Week8_Final_Analytics_Dashboard.html
- documentation/Week8_Integration_Readiness_and_Cross_Track_Evidence.md
- documentation/Individual_Video_Presentation_Script.md
- social/LinkedIn_and_X_Drafts.md
- data/raw/ unchanged source dataset and data dictionary copy

## Reproduce the analysis

From this package root, use Python with pandas and numpy installed:

    python analytics/src/analyze_healthconnect.py

The script regenerates the KPI, segment and data-quality CSVs, plus four charts under analytics/outputs/. The notebook explains KPI definitions and source checks.

## KPI definitions

- No-show rate = No-Show / (Attended + No-Show).
- Attendance rate = Attended / (Attended + No-Show).
- Cancellation rate = Cancelled / all appointment records.
- Reminder coverage = Reminder Sent = Yes / all appointment records.

Cancellations remain separate from completed appointments. Patient IDs repeat, so appointments are not independent people. The reminder comparison is descriptive and does not estimate an intervention effect.

## Readiness and limitations

The analytics component is ready for presentation as descriptive decision support. It is not a deployed system or clinical decision tool. The data are synthetic, observed associations are not causal, repeated records limit person-level inference, and the Week 7 Analytics handoff has not been verified.

## Source integrity

Raw source SHA-256: 1f2c40ae14833b46fe6e1a353446a0e038f19ae06434b37d98545b3e28697137
