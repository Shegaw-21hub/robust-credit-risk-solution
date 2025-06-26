# Credit Risk Modeling Project

## Credit Scoring Business Understanding

### 1. Basel II Accord and Model Interpretability

The Basel II Accord emphasizes three pillars: minimum capital requirements, supervisory review, and market discipline. For our model:
- **Regulatory Compliance**: The accord requires banks to demonstrate that their risk measurement systems are conceptually sound and empirically validated. This necessitates an interpretable model where decision-making can be clearly explained to regulators.
- **Documentation**: We must thoroughly document our modeling process, including feature selection, proxy variable creation, and validation procedures.
- **Risk Sensitivity**: The model must accurately differentiate risk levels to ensure proper capital allocation according to risk weights.

### 2. Proxy Variable Necessity and Risks

Since we lack direct default data:
- **Necessity**: A proxy based on RFM (Recency, Frequency, Monetary) metrics allows us to approximate credit risk from transactional behavior patterns that correlate with repayment likelihood.
- **Business Risks**:
  - **Misclassification Risk**: The proxy might mislabel some customers, leading to either lost revenue (false positives) or increased defaults (false negatives).
  - **Concept Drift**: Behavioral patterns in e-commerce may not perfectly correlate with credit repayment behavior.
  - **Regulatory Scrutiny**: We must justify our proxy methodology to satisfy compliance requirements.

### 3. Model Complexity Trade-offs

**Simple Models (Logistic Regression with WoE)**:
- *Advantages*: Easily interpretable, compliant with "right to explanation" regulations, simpler to validate and audit.
- *Disadvantages*: May miss complex nonlinear relationships, potentially lower predictive power.

**Complex Models (Gradient Boosting)**:
- *Advantages*: Higher predictive accuracy, can capture intricate feature interactions.
- *Disadvantages*: "Black box" nature raises regulatory concerns, harder to explain decisions to customers.

**Recommended Approach**: Start with interpretable models and only increase complexity if justified by significant performance gains that outweigh regulatory costs.