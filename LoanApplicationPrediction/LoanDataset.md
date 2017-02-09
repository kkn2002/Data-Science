Gain an understanding of a small loan-application dataset so that we can double-check the work of agents making quick-fire loan determinations by text.




Goals


1. Data Understanding
 - track any statistics or findings that are relevant to solving the problem.
    · 
2. Provide a thorough analysis of the relationships between the target and the various features and also the relationships between the important features.

3. Prediction / Classification - Build a quick classifier to determine what the result should be for a new customer
    
4. Explain the results and the reason(s) the model is valid


Data

ds-credit: CustomerID CheckingAccountBalance DebtsPaid SavingsAccountBalance CurrentOpenLoanApplications

ds-app: CustomerID LoanPayoffPeriodInMonths LoanReason RequestedAmount InterestRate Co-Applicant

ds-borrower: CustomerID YearsAtCurrentEmployer YearsInCurrentResidence Age RentOrOwnHome TypeOfCurrentEmployment NumberOfDependantsIncludingSelf

ds-result: CustomerID WasTheLoanApproved
