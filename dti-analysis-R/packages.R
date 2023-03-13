# Packages
install.packages('lavaan', dependencies = TRUE, INSTALL_opts = '--no-lock')
install.packages('psych', type = 'source', repos = 
                   'http://personality-project.org/r/')
install.packages(("GPArotation"))
install.packages('semTools', INSTALL_opts = '--no-lock')

library('lavaan')
library('psych')
library('GPArotation')
library('semTools')
