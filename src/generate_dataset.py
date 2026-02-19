import pandas as pd
import numpy as np

def generate_data(n=1000):
    np.random.seed(42)

    loan_amount = np.random.randint(20000, 300000, n)
    income = np.random.randint(30000, 150000, n)
    credit_score = np.random.randint(300, 850, n)
    loan_term = np.random.choice([12, 24, 36, 48, 60, 72], n)
    existing_loans = np.random.randint(0, 5, n)

    # Risk logic (realistic pattern)
    default = (
        (credit_score < 600).astype(int) |
        ((income < 50000) & (loan_amount > 150000)).astype(int) |
        (existing_loans > 3).astype(int)
    )

    df = pd.DataFrame({
        "loan_amount": loan_amount,
        "income": income,
        "credit_score": credit_score,
        "loan_term": loan_term,
        "existing_loans": existing_loans,
        "default": default
    })

    df.to_csv("data/loan_data.csv", index=False)
    print("Dataset generated successfully!")

if __name__ == "__main__":
    generate_data()
