"""
init_db.py
-----------
Modular initialization script for the AWS Mock Exam Engine database.

Running this script will:
  1. Create aws_exams.db (if it does not exist) with the required schema.
  2. Seed the five target certifications.
  3. Seed one high-quality, exam-realistic sample question per certification
     (idempotent — running multiple times will not create duplicate seed
     questions because we check before inserting).

Usage:
    python init_db.py
"""

import sqlite3
import db_utils

# One realistic, professional-grade sample question per certification.
# `exam_number` = 1 marks these as belonging to "Practice Exam #1" for that cert.
# The schema supports up to 10 exam_number values (1-10) per certification,
# each holding ~65-75 questions, for full-length exam simulation down the line.
SEED_QUESTIONS = {
    "SAP-C02": {
        "question_text": (
            "A company runs a multi-tier application across three AWS Regions for disaster "
            "recovery. The database tier uses Amazon Aurora MySQL with a Global Database. "
            "During a recent regional outage, the failover to the secondary Region took over "
            "20 minutes and required manual intervention, breaching the company's RTO of 5 "
            "minutes. A solutions architect must redesign the failover process to meet the RTO "
            "while minimizing operational overhead. Which solution meets these requirements?"
        ),
        "option_a": (
            "Enable Aurora Global Database managed planned failover and trigger it manually "
            "using the AWS CLI whenever CloudWatch alarms indicate a regional outage."
        ),
        "option_b": (
            "Implement Amazon Route 53 health checks against the primary Region's endpoint "
            "combined with an AWS Lambda function that promotes the secondary Aurora Global "
            "Database cluster automatically, and use Route 53 failover routing to redirect "
            "traffic to the secondary Region."
        ),
        "option_c": (
            "Replace Aurora Global Database with cross-Region read replicas and configure "
            "an Application Load Balancer with cross-zone load balancing to route traffic to "
            "the closest healthy replica."
        ),
        "option_d": (
            "Reduce the RTO requirement to 20 minutes and document the manual failover runbook "
            "so operations staff can execute it faster during future incidents."
        ),
        "correct_option": "B",
        "explanation": (
            "Aurora Global Database supports a managed failover process, but for unplanned "
            "regional outages the fastest, least operationally burdensome pattern is to "
            "automate detection and promotion: Route 53 health checks detect the primary "
            "Region's unavailability, a Lambda function programmatically promotes the "
            "secondary Aurora cluster (removing manual CLI steps), and Route 53 failover "
            "routing shifts application traffic to the newly promoted Region — all without "
            "human intervention, which is required to reliably meet a 5-minute RTO. Option A "
            "still requires a human to trigger the failover, which cannot reliably meet a "
            "5-minute RTO. Option C removes Global Database's fast promotion capability and "
            "replaces it with standard read replicas, which have higher replication lag and "
            "no built-in promotion tooling. Option D does not solve the underlying problem; "
            "it just changes the requirement rather than engineering the solution."
        ),
        "domain": "Design for New Solutions / High Availability",
    },
    "DOP-C02": {
        "question_text": (
            "A company uses AWS CodePipeline to deploy a containerized application to Amazon "
            "ECS. The DevOps team wants every production deployment to automatically roll back "
            "if the new task set generates elevated 5xx errors or high latency within the "
            "first 10 minutes after deployment, without requiring manual monitoring. Which "
            "combination of actions should the DevOps engineer implement? (Select the BEST "
            "single answer.)"
        ),
        "option_a": (
            "Configure the ECS service to use a rolling update deployment type and manually "
            "monitor Amazon CloudWatch dashboards for 10 minutes after each deployment, rolling "
            "back via the ECS console if error rates spike."
        ),
        "option_b": (
            "Configure the ECS service deployment controller to use AWS CodeDeploy with a "
            "blue/green deployment configuration, define CloudWatch alarms on 5xx error rate "
            "and latency metrics, and associate those alarms with the CodeDeploy deployment "
            "group so that CodeDeploy automatically rolls back the deployment if an alarm "
            "enters ALARM state during the specified bake time."
        ),
        "option_c": (
            "Use an AWS Lambda function triggered on a CloudWatch Events schedule every minute "
            "to query CloudWatch metrics and manually invoke the ECS UpdateService API to "
            "revert to the previous task definition if thresholds are breached."
        ),
        "option_d": (
            "Enable AWS CodePipeline's built-in automatic rollback feature, which reverts to "
            "the previous pipeline execution whenever any CloudWatch alarm in the account "
            "enters ALARM state."
        ),
        "correct_option": "B",
        "explanation": (
            "CodeDeploy blue/green deployments for ECS natively support automatic rollback "
            "driven by CloudWatch alarms during a configurable bake time (deployment monitoring "
            "period). By attaching alarms for 5xx error rate and latency to the deployment "
            "group, CodeDeploy will automatically stop routing traffic to the new task set and "
            "roll back to the last known-good version with zero manual intervention — precisely "
            "meeting the requirement. Option A relies on manual monitoring, which does not meet "
            "the 'without manual monitoring' requirement. Option C reinvents functionality that "
            "CodeDeploy already provides natively and introduces unnecessary custom code and "
            "operational risk. Option D describes functionality that does not exist as a native, "
            "generic CodePipeline feature tied arbitrarily to any account-wide alarm."
        ),
        "domain": "Configuration Management and Infrastructure as Code / Resilient Cloud Solutions",
    },
    "MLS-C01": {
        "question_text": (
            "A data scientist is training a binary classification model in Amazon SageMaker to "
            "detect fraudulent credit card transactions. The training dataset is highly "
            "imbalanced: fraudulent transactions make up only 0.5% of all records. After "
            "training an XGBoost model, the model achieves 99.5% accuracy on the validation set "
            "but fails to identify almost any actual fraud cases. Which combination of steps "
            "should the data scientist take to BEST improve the model's ability to detect fraud?"
        ),
        "option_a": (
            "Increase the number of training epochs and add more decision trees to the XGBoost "
            "model to increase overall accuracy."
        ),
        "option_b": (
            "Apply a technique such as SMOTE (Synthetic Minority Over-sampling Technique) or "
            "class-weighted training to address class imbalance, and evaluate the model using "
            "precision, recall, and the F1 score (or AUC-PR) instead of accuracy."
        ),
        "option_c": (
            "Remove all non-fraudulent transactions from the training dataset so the classes "
            "are represented equally, then evaluate the model using overall classification "
            "accuracy on the modified dataset."
        ),
        "option_d": (
            "Switch from a supervised XGBoost model to an unsupervised k-means clustering "
            "algorithm, since clustering does not require labeled fraud data."
        ),
        "correct_option": "B",
        "explanation": (
            "With a severely imbalanced dataset (0.5% positive class), accuracy is a misleading "
            "metric because a model that always predicts 'not fraud' would still score 99.5% "
            "accuracy. The correct approach is to address the imbalance directly — via "
            "oversampling techniques like SMOTE, undersampling, or class-weighted loss functions "
            "(e.g., XGBoost's scale_pos_weight parameter) — and then evaluate using metrics that "
            "reflect performance on the minority class, such as precision, recall, F1 score, or "
            "area under the precision-recall curve (AUC-PR). Option A does nothing to address "
            "class imbalance and will likely worsen overfitting to the majority class. Option C "
            "discards the vast majority of legitimate transaction data, destroying valuable "
            "signal and making the model unrepresentative of real-world traffic. Option D "
            "abandons the labeled data entirely, which is unnecessary and generally "
            "underperforms supervised methods when good labels already exist."
        ),
        "domain": "Modeling / Evaluation",
    },
    "AIF-C01": {
        "question_text": (
            "A company wants to build a generative AI application that answers customer "
            "questions using the company's internal product documentation, without exposing "
            "that documentation to the foundation model's training process and without needing "
            "to fine-tune the model. Which approach BEST meets these requirements?"
        ),
        "option_a": (
            "Fine-tune a foundation model in Amazon Bedrock directly on the internal product "
            "documentation so the model memorizes the content."
        ),
        "option_b": (
            "Use Retrieval Augmented Generation (RAG) by storing the documentation as embeddings "
            "in a vector database and retrieving relevant chunks at query time to include as "
            "context in the prompt sent to the foundation model."
        ),
        "option_c": (
            "Manually copy and paste the entire documentation set into the system prompt for "
            "every single user query, regardless of the question asked."
        ),
        "option_d": (
            "Train a completely new foundation model from scratch using only the company's "
            "product documentation as training data."
        ),
        "correct_option": "B",
        "explanation": (
            "Retrieval Augmented Generation (RAG) is the standard pattern for grounding a "
            "foundation model's responses in proprietary or frequently-changing data without "
            "modifying the model's weights. Relevant document chunks are retrieved from a "
            "vector database based on semantic similarity to the user's query and injected into "
            "the prompt as context, so the model's underlying training data is never touched and "
            "no fine-tuning is required. Option A requires fine-tuning, which the requirements "
            "explicitly exclude, and also risks the documentation being absorbed into model "
            "weights. Option C does not scale, wastes context window space on irrelevant "
            "content, and increases cost and latency for every request. Option D is prohibitively "
            "expensive, requires enormous datasets, and is unnecessary for this use case."
        ),
        "domain": "Fundamentals of Generative AI",
    },
    "CLF-C02": {
        "question_text": (
            "A company is evaluating AWS Cloud services and wants to understand which AWS "
            "pricing model would provide the LOWEST cost for a workload that must run "
            "continuously, 24/7, for the next 3 years, and where the instance type is not "
            "expected to change. Which purchasing option should the company choose?"
        ),
        "option_a": "On-Demand Instances, paying by the second with no long-term commitment.",
        "option_b": (
            "A 3-year Reserved Instance (or Savings Plan) with the All Upfront payment option."
        ),
        "option_c": (
            "Spot Instances, which bid on unused EC2 capacity at up to a 90% discount but can "
            "be interrupted with short notice."
        ),
        "option_d": (
            "Dedicated Hosts billed on a per-second basis with no capacity reservation."
        ),
        "correct_option": "B",
        "explanation": (
            "For a steady-state, predictable workload running continuously for a known, long "
            "duration (3 years), a 3-year Reserved Instance or Savings Plan with All Upfront "
            "payment provides the deepest discount compared to On-Demand pricing — typically the "
            "lowest effective hourly cost of all EC2 purchasing options for this exact use case. "
            "On-Demand (Option A) offers maximum flexibility but at the highest cost, which is "
            "wasteful for a workload with no variability. Spot Instances (Option C) offer steep "
            "discounts but are unsuitable here because the workload must run continuously and "
            "Spot capacity can be reclaimed by AWS with only a two-minute warning, risking "
            "interruption of a required always-on workload. Dedicated Hosts (Option D) address "
            "licensing and compliance needs for dedicated physical servers but are not the "
            "lowest-cost option for this scenario."
        ),
        "domain": "Billing, Pricing, and Support",
    },
}


def question_already_seeded(conn: sqlite3.Connection, cert_id: int, question_text: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM questions WHERE cert_id = ? AND question_text = ?",
        (cert_id, question_text),
    ).fetchone()
    return row is not None


def seed_sample_questions(db_path: str = db_utils.DB_PATH) -> None:
    """Idempotently insert one realistic sample question per certification."""
    with db_utils.get_connection(db_path) as conn:
        for code, q in SEED_QUESTIONS.items():
            cert = conn.execute(
                "SELECT id FROM certifications WHERE code = ?", (code,)
            ).fetchone()
            if cert is None:
                print(f"  [WARN] Certification {code} not found — skipping seed question.")
                continue
            if question_already_seeded(conn, cert["id"], q["question_text"]):
                print(f"  [SKIP] Sample question for {code} already exists.")
                continue
            conn.execute(
                """
                INSERT INTO questions
                    (cert_id, exam_number, question_text, option_a, option_b, option_c, option_d,
                     correct_option, explanation, domain)
                VALUES (?, 1, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    cert["id"], q["question_text"], q["option_a"], q["option_b"],
                    q["option_c"], q["option_d"], q["correct_option"], q["explanation"],
                    q["domain"],
                ),
            )
            print(f"  [OK] Seeded sample question for {code}.")


def main():
    print("Initializing AWS Mock Exam Engine database...")
    db_utils.init_schema()
    print("  [OK] Schema created/verified.")
    db_utils.seed_certifications()
    print("  [OK] Certifications seeded/verified.")
    seed_sample_questions()

    print("\nSummary:")
    for row in db_utils.get_question_counts_all():
        print(f"  {row['code']:10s} {row['name']:55s} -> {row['question_count']} question(s)")
    print("\nDatabase ready: aws_exams.db")


if __name__ == "__main__":
    main()
