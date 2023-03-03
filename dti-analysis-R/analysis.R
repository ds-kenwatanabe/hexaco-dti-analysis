df=read.csv('dti.csv')

DTI.model <- ' preference_for_dichotomy =~ dti_1 + dti_4 + dti_7 + 
              dti_10 + dti_13
              profit_loss_thinking =~ dti_2 + dti_5 + dti_8 + 
              dti_11 + dti_14
              dichotomous_belief =~ dti_3 + dti_6 + dti_9 + 
              dti_12 + dti_15 '

fit <- cfa(DTI.model, data = df)

summary(fit, fit.measures=TRUE)

polychoric(df)

# Parallel analysis
scree(df)
fa.parallel(df, nfactors=4, fm='ml', fa='fa', 
            main='Parallel Analysis Scree Plots', 
            n.iter=1000)

# VSS, MAP
vss(df, n=4, fm="mle", rotate="oblimin")


# 2 new factors
FACTORS2.model <- ' factor_1 =~ dti_1 + dti_2 + dti_3 + 
              dti_4 + dti_5 + dti_6 + dti_7 + dti_8 + 
              dti_9 + dti_10
              factor_2 =~ dti_11 + dti_12 + dti_13 + 
              dti_14 + dti_15'

fit2 <- cfa(FACTORS2.model, data = df)
summary(fit2, fit.measures=TRUE)

# 3 new factors
FACTORS3.model <- ' factor_1 =~ dti_1 + dti_2 + dti_3 + 
              dti_4 + dti_5
              factor_2 =~ dti_6 + dti_7 + dti_8 + 
              dti_9 + dti_10
              factor_3 =~ dti_11 + dti_12 + dti_13 + 
              dti_14 + dti_15 '

fit3 <- cfa(FACTORS3.model, data = df)
summary(fit3, fit.measures=TRUE)

# 4 new factors
FACTORS4.model <- ' factor_1 =~ dti_1 + dti_2 + dti_3 + 
              dti_4 + dti_5
              factor_2 =~ dti_6 + dti_7 + dti_8 + 
              dti_9 + dti_10
              factor_3 =~ dti_13 + dti_14 + dti_15
              factor_4 =~ dti_11 + dti_12 '

fit4 <- cfa(FACTORS4.model, data = df)
summary(fit4, fit.measures=TRUE)
