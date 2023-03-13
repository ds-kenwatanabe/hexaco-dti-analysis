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
              factor_2 =~ dti_11 + dti_12 + 
              dti_13 + dti_14 + dti_15'

fit2 <- cfa(FACTORS2.model, data = df, std.lv=TRUE, estimator='WLSMV')
summary(fit2, fit.measures=TRUE)

# Calculating reliability
semTools::reliability(fit2)

# Residuals
residuals(fit2, type='cor')

# new model accounting for residual correlation
mod2.model =  'factor_1 =~ dti_1 + dti_2 + dti_3 + 
              dti_4 + dti_5 + dti_6 + dti_7 + dti_8 + 
              dti_9 + dti_10
              factor_2 =~ dti_11 + dti_12 + 
              dti_13 + dti_14 + dti_15
              dti_2 ~~ dti_1
              dti_3 ~~ dti_1
              dti_5 ~~ dti_1
              dti_15 ~~ dti_1
              dti_3 ~~ dti_2
              dti_5 ~~ dti_2
              dti_12 ~~ dti_2
              dti_8 ~~ dti_6
              dti_12 ~~ dti_6
              dti_14 ~~ dti_7
              dti_11 ~~ dti_9
              dti_12 ~~ dti_9
              dti_13 ~~ dti_9
              dti_12 ~~ dti_11
              dti_15 ~~ dti_11
              dti_15 ~~ dti_12'
fitmod <- cfa(mod2.model, data = df, std.lv=TRUE, estimator='WLSMV')
summary(fitmod, fit.measures=TRUE)

# Residuals
residuals(fitmod, type='cor')

# Calculating reliability
semTools::reliability(fitmod)

# Bi-factor model with general factor
modgen.model =  'gen =~ dti_1 + dti_2 + dti_3 + 
              dti_4 + dti_5 + dti_6 + dti_7 + dti_8 + 
              dti_9 + dti_10 + dti_11 + dti_12 + 
              dti_13 + dti_14 + dti_15
              factor_1 =~ dti_1 + dti_2 + dti_3 + 
              dti_4 + dti_5 + dti_6 + dti_7 + dti_8 + 
              dti_9 + dti_10
              factor_2 =~ dti_11 + dti_12 + 
              dti_13 + dti_14 + dti_15'
fitmodgen <- cfa(modgen.model, data = df, std.lv=TRUE, orthogonal=TRUE, 
                 estimator='WLSMV')
summary(fitmodgen, fit.measures=TRUE)

# Calculating reliability
semTools::reliability(fitmodgen)

# Omega
omega(df, nfactors = 2, fm = "ml")

# 1 factor
FACTORS1.model <- ' factor_1 =~ dti_1 + dti_2 + dti_3 + 
              dti_4 + dti_5 + dti_6 + dti_7 + dti_8 + 
              dti_9 + dti_10 + dti_11 + dti_12 + 
              dti_13 + dti_14 + dti_15'

fit1 <- cfa(FACTORS1.model, data = df, std.lv=TRUE)
summary(fit1, fit.measures=TRUE)

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
