df=read.csv('dti.csv')
df_fact=read.csv('df_fact.csv')
DTI.model <- ' preference_for_dichotomy =~ dti_1 + dti_4 + dti_7 + 
              dti_10 + dti_13
              profit_loss_thinking =~ dti_2 + dti_5 + dti_8 + 
              dti_11 + dti_14
              dichotomous_belief =~ dti_3 + dti_6 + dti_9 + 
              dti_12 + dti_15 '

fit <- cfa(DTI.model, data = df)

summary(fit, fit.measures=TRUE)

polychoric(df)

FACTORS.model <- ' factor_1 =~ dti_1 + dti_2 + dti_3 + 
              dti_4 + dti_5
              factor_2 =~ dti_6 + dti_7 + dti_8 + 
              dti_9 + dti_10
              factor_3 =~ dti_11 + dti_12 + dti_13 + 
              dti_14 + dti_15 '

fit2 <- cfa(FACTORS.model, data = df)
summary(fit2, fit.measures=TRUE)
