# Iris DB Generator
This code contains random iris data generation and database encryption for CCMM, which is required for the core component.
* The plain iris templates are stored in `../../random_data`.
* The encrypted database is stored in `../../encrypted_db` together with its secret keys.

The query templates contain both matching and non-matching ones, so the coreCircuit checks both matching and non-matching cases in a batch.