import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pingouin as pg
import scipy.stats as stats
import statsmodels.api as sm

df = pd.read_csv('pesquisa-mestrado-chris_February+1,+2023_16.29.csv')

# Dropping first two rows (import id)
df = df.drop(df.index[[0, 1]])

# Changing courses to major areas of knowledge
df.loc[df["course"] == "Outro", "course"] = df[df["course"] == "Outro"]["course2"]

course_names = df['course'].value_counts().index.tolist()
# print(course_names)

# Dropping course2 column and NaN values
df = df.drop(['course2'], axis=1)
df = df.dropna()

# Typecasting age
df['age'] = df['age'].astype('int')

# Replacing course names for areas of knowledge
cols = ["course"]
replace_dict = {"Direito": "Ciências Humanas", 'Psicologia': 'Ciências Biológicas',
                'Medicina': 'Ciências Biológicas',
                'Geografia': "Ciências Humanas", 'Economia': "Ciências Humanas",
                'Engenharia Elétrica e de Computação': 'Ciências Exatas',
                'Geologia': 'Ciências Exatas', 'Design': "Ciências Humanas",
                'Arquitetura': "Ciências Humanas", 'Engenharia Química': 'Ciências Exatas',
                'Enfermagem': 'Ciências Biológicas', 'Computação': 'Ciências Exatas',
                'Relações Internacionais': 'Ciências Humanas',
                'Educação Física': 'Ciências Biológicas',
                'Nutrição': 'Ciências Biológicas', 'Medicina Veterinária': 'Ciências Biológicas',
                'Letras': 'Ciências Humanas', 'Terapia Ocupacional': 'Ciências Biológicas',
                'Engenharia de Produção': 'Ciências Exatas',
                'Ciências Contábeis': 'Ciências Exatas', 'História': 'Ciências Humanas',
                'Biblioteconomia': 'Ciências Humanas', 'Ciência da Informação': 'Ciências Exatas',
                'Artes Visuais': 'Ciências Humanas', 'Pedagogia': 'Ciências Humanas',
                'Administração': 'Ciências Humanas', 'Física': 'Ciências Exatas',
                'Engenharia Civil': 'Ciências Exatas', 'Farmácia': 'Ciências Biológicas',
                'Engenharia Agronômica': 'Ciências Exatas', 'Matemática': 'Ciências Exatas',
                'Arqueologia ': 'Ciências Humanas', 'Química': 'Ciências Exatas',
                'Ciências Sociais': 'Ciências Humanas',
                'Sistemas de Informação': 'Ciências Exatas',
                'Ciências Biomédicas': 'Ciências Biológicas',
                'Filosofia': 'Ciências Humanas', 'Gestão de Políticas Públicas': 'Ciências Humanas',
                'Fisioterapia': 'Ciências Biológicas', 'Biblioteconomia ': 'Ciências Humanas',
                'Turismo': 'Ciências Humanas', 'Engenharia Ambiental': 'Ciências Exatas',
                'Ciências Atuariais': 'Ciências Humanas', 'Engenharia Mecânica': 'Ciências Exatas',
                'Arqueologia': 'Ciências Humanas', 'Biotecnologia': 'Ciências Biológicas',
                'Ciências Atuariais ': 'Ciências Humanas', 'Jornalismo': 'Ciências Humanas',
                'Engenharia Naval': 'Ciências Exatas', 'Engenharia de Software ': 'Ciências Exatas',
                'Psicologia ': 'Ciências Biológicas', 'Engenharia de Alimentos': 'Ciências Exatas',
                'Publicidade e Propaganda': 'Ciências Humanas', 'Gestão Ambiental': 'Ciências Humanas',
                'Odontologia': 'Ciências Biológicas', 'Saúde Pública': 'Ciências Biológicas',
                'Serviço Social ': 'Ciências Humanas',
                'Serviço Social': 'Ciências Humanas', 'Engenharia de Sistemas': 'Ciências Exatas',
                'Secretariado Executivo': 'Ciências Humanas', 'Licenciatura em geografia ': 'Ciências Humanas',
                'Teologia ': 'Ciências Humanas', 'engenharia de software ': 'Ciências Exatas',
                'Estudo de Midias/Audiovisual ': 'Ciências Humanas', 'Zootecnia': 'Ciências Biológicas',
                'Educação Física ': 'Ciências Biológicas', 'Engenharia Bioquímica': 'Ciências Exatas',
                'Línguas Estrangeiras Aplicadas ': 'Ciências Humanas',
                'Geografia - Licenciatura ': 'Ciências Humanas',
                'engenharia de sistemas': 'Ciências Exatas', 'Museologia': 'Ciências Humanas',
                'Sociologia': 'Ciências Humanas', 'Ciências do Estado': 'Ciências Humanas',
                'Bioquímica ': 'Ciências Exatas',
                'Licenciatura interdisciplinar em ciências humanas e sociais e suas tecnologias ': 'Ciências Humanas',
                'Ciencias atuariais': 'Ciências Humanas', 'Ciências da computação': 'Ciências Exatas',
                'Licenciatura em ciências exatas': 'Ciências Exatas',
                'Engenharia de Software': 'Ciências Exatas',
                'Música': 'Ciências Humanas',
                'Engenharia de Bioprocessos e Biotecnologia ': 'Ciências Exatas',
                'Bacharelado em Ciências e Humanidades - UFABC': 'Ciências Humanas',
                'psicologia ': 'Ciências Humanas', 'Medicina veterinária ': 'Ciências Humanas',
                'Saude coletiva': 'Ciências Biológicas', 'Linguística ': 'Ciências Humanas',
                'Quiropraxia ': 'Ciências Biológicas', 'Ciências ambientais ': 'Ciências Exatas',
                'Estatística ': 'Ciências Exatas',
                'Engenharia Bioprocessos e Biotecnologia ': 'Ciências Exatas',
                'Gastronomia ': 'Ciências Humanas', 'Relações Internacionais ': 'Ciências Humanas',
                'Estatística': 'Ciências Exatas', 'Radio, TV e Internet ': 'Ciências Humanas',
                'Serviço social ': 'Ciências Humanas', 'Matemática Aplicada': 'Ciências Exatas',
                'Geociências': 'Ciências Exatas',
                'Licenciatura em Ciências Naturais e Meio Ambiente': 'Ciências Biológicas',
                'Licenciatura em Geociências e Educação Ambiental': 'Ciências Exatas',
                'Rádio, TV e Internet': 'Ciências Humanas',
                'ARQUEOLOGIA': 'Ciências Humanas',
                'Análise e Desenvolvimento de Sistemas': 'Ciências Exatas',
                'Teologia': 'Ciências Humanas',
                'Fisioterapia, Educação do Campo ciências da natureza': 'Ciências Biológicas',
                'Engenharia Mecatrônica': 'Ciências Exatas',
                'Biomedicina ': 'Ciências Biológicas',
                'Licenciatura em Geociências e Gestão Ambiental': 'Ciências Exatas',
                'Ciência de Dados': 'Ciências Exatas',
                'Licenciatura em Geociências e educação ambiental': 'Ciências Exatas',
                'Editoração': 'Ciências Humanas'}

for col in cols:
    df[col] = df[col].replace(replace_dict)

# Replacing the three branches of sciences names to english
cols2 = ['course']
course_dict = {'Ciências Biológicas': 'Biological Sciences',
               'Ciências Exatas': 'Exact Sciences',
               'Ciências Humanas': 'Social Sciences'}
for column in cols2:
    df[column] = df[column].replace(course_dict)

# Changing course semester to numerical values
df['semester'] = df.semester.replace({'1º': 1, '2º': 2, '3º': 3, '4º': 4, '5º': 5,
                                      '6º': 6, '7º': 7, '8º': 8, '9º': 9,
                                      '10º': 10, '11º': 11, '12º': 12})

# Excluding sex other than male/female
df = df[df['sex'] != 'Outro (especifique)']

# Replacing sex categories to english
df['sex'] = df['sex'].replace({'Masculino': 'Male', 'Feminino': 'Female'})

# Excluding asexuals
df = df[df['kinsey'] != 'Não tenho atração por nenhum gênero']

# Changing Kinsey scale to heterosexual, bi and homo
df['kinsey'] = df['kinsey'].replace({'0 - Exclusivamente heterossexual': 'Heterosexual',
                                     '1 - Predominantemente heterossexual, '
                                     'mas incidentalmente homossexual': 'Heterosexual',
                                     '2 - Predominantemente heterossexual, entretanto, '
                                     'mais que incidentalmente homossexual': 'Bisexual',
                                     '3 - Igualmente heterossexual e homossexual': 'Bisexual',
                                     '4 - Predominantemente homossexual, entretanto, '
                                     'mais que incidentalmente heterossexual': 'Bisexual',
                                     '5 - Predominantemente homossexual, '
                                     'mas incidentalmente heterossexual': 'Homosexual',
                                     '6 - Exclusivamente homossexual': 'Homosexual'})

# Indicator/Dummy coded variables
df['sex_num'] = df.sex.replace({'Male': 1, 'Female': 2})
df['course_num'] = df.course.replace({'Social Sciences': 2,
                                      'Biological Sciences': 1,
                                      'Exact Sciences': 3})
df['kinsey_num'] = df.kinsey.replace({'Heterosexual': 1, 'Bisexual': 2, 'Homosexual': 3})

# Total DTI
# Changing DTI values to numerical values
dti_dict = {'1 - Discordo Totalmente': 1, '2 - Discordo': 2,
            '3 - Discordo um Pouco': 3, '4 - Concordo um Pouco': 4,
            '5 - Concordo': 5, '6 - Concordo Totalmente': 6}

dti = ['dti_1', 'dti_4', 'dti_7', 'dti_10', 'dti_13',
       'dti_2', 'dti_5', 'dti_8', 'dti_11', 'dti_14',
       'dti_3', 'dti_6', 'dti_9', 'dti_12', 'dti_15']

df[dti] = df[dti].replace(dti_dict)
df['dti'] = df[dti].sum(axis=1)

# Dividing into subscales scores
preference_for_dichotomy = ['dti_1', 'dti_4', 'dti_7', 'dti_10', 'dti_13']
df['preference_for_dichotomy'] = df[preference_for_dichotomy].sum(axis=1)

dichotomous_belief = ['dti_2', 'dti_5', 'dti_8', 'dti_11', 'dti_14']
df['dichotomous_belief'] = df[dichotomous_belief].sum(axis=1)

profit_loss_thinking = ['dti_3', 'dti_6', 'dti_9', 'dti_12', 'dti_15']
df['profit_loss_thinking'] = df[profit_loss_thinking].sum(axis=1)

# Changing HEXACO values to numerical values
hexaco_dict = {'Discordo fortemente': 1, 'Discordo': 2,
               'Neutro': 3, 'Concordo': 4, 'Concordo fortemente': 5}

hexaco_dict_reverse = {'Discordo fortemente': 5,
                       'Discordo': 4, 'Neutro': 3,
                       'Concordo': 2, 'Concordo fortemente': 1}

# df['hexaco_1'] = df['hexaco_1'].apply(lambda x: hexaco_dict_reverse[x])

# Reversed and regular coded HEXACO values
reverse = ['hexaco_1', 'hexaco_19', 'hexaco_31', 'hexaco_49', 'hexaco_55',
           'hexaco_14', 'hexaco_20', 'hexaco_26', 'hexaco_32', 'hexaco_44',
           'hexaco_56', 'hexaco_9', 'hexaco_15', 'hexaco_21', 'hexaco_57',
           'hexaco_10', 'hexaco_28', 'hexaco_46', 'hexaco_52', 'hexaco_35',
           'hexaco_41', 'hexaco_53', 'hexaco_59', 'hexaco_12', 'hexaco_24',
           'hexaco_30', 'hexaco_42', 'hexaco_48', 'hexaco_60']

regular = ['hexaco_7', 'hexaco_13', 'hexaco_25', 'hexaco_37', 'hexaco_43',
           'hexaco_2', 'hexaco_8', 'hexaco_38', 'hexaco_50', 'hexaco_3',
           'hexaco_27', 'hexaco_33', 'hexaco_39', 'hexaco_45', 'hexaco_51',
           'hexaco_4', 'hexaco_16', 'hexaco_22', 'hexaco_34', 'hexaco_40',
           'hexaco_58', 'hexaco_5', 'hexaco_11', 'hexaco_17', 'hexaco_23',
           'hexaco_29', 'hexaco_47', 'hexaco_6', 'hexaco_18', 'hexaco_36',
           'hexaco_54']

df[regular] = df[regular].replace(hexaco_dict)
df[reverse] = df[reverse].replace(hexaco_dict_reverse)

# Separating by HEXACO traits means
H = ['hexaco_6', 'hexaco_12', 'hexaco_18', 'hexaco_24', 'hexaco_30',
     'hexaco_36', 'hexaco_42', 'hexaco_48', 'hexaco_54', 'hexaco_60']
df['H'] = df[H].mean(axis=1)

E = ['hexaco_5', 'hexaco_11', 'hexaco_17', 'hexaco_23', 'hexaco_29',
     'hexaco_35', 'hexaco_41', 'hexaco_47', 'hexaco_53', 'hexaco_59']
df['E'] = df[E].mean(axis=1)

X = ['hexaco_4', 'hexaco_10', 'hexaco_16', 'hexaco_22', 'hexaco_28',
     'hexaco_34', 'hexaco_40', 'hexaco_46', 'hexaco_52', 'hexaco_58']
df['X'] = df[X].mean(axis=1)

A = ['hexaco_3', 'hexaco_9', 'hexaco_15', 'hexaco_21', 'hexaco_27',
     'hexaco_33', 'hexaco_39', 'hexaco_45', 'hexaco_51', 'hexaco_57']
df['A'] = df[A].mean(axis=1)

C = ['hexaco_2', 'hexaco_8', 'hexaco_14', 'hexaco_20', 'hexaco_26',
     'hexaco_32', 'hexaco_38', 'hexaco_44', 'hexaco_50', 'hexaco_56']
df['C'] = df[C].mean(axis=1)

O = ['hexaco_1', 'hexaco_7', 'hexaco_13', 'hexaco_19', 'hexaco_25',
     'hexaco_31', 'hexaco_37', 'hexaco_43', 'hexaco_49', 'hexaco_55']
df['O'] = df[O].mean(axis=1)

# HEXACO facets
# H
H_sincerity = ['hexaco_6', 'hexaco_30', 'hexaco_54']
df['H_sincerity'] = df[H_sincerity].mean(axis=1)

H_fairness = ['hexaco_12', 'hexaco_36', 'hexaco_60']
df['H_fairness'] = df[H_fairness].mean(axis=1)

H_greed_avoidance = ['hexaco_18', 'hexaco_42']
df['H_greed_avoidance'] = df[H_greed_avoidance].mean(axis=1)

H_modesty = ['hexaco_24', 'hexaco_48']
df['H_modesty'] = df[H_modesty].mean(axis=1)

# E
E_fearfulness = ['hexaco_5', 'hexaco_29', 'hexaco_53']
df['E_fearfulness'] = df[E_fearfulness].mean(axis=1)

E_anxiety = ['hexaco_11', 'hexaco_35']
df['E_anxiety'] = df[E_anxiety].mean(axis=1)

E_dependence = ['hexaco_17', 'hexaco_41']
df['E_dependence'] = df[E_dependence].mean(axis=1)

E_sentimentality = ['hexaco_23', 'hexaco_47', 'hexaco_59']
df['E_sentimentality'] = df[E_sentimentality].mean(axis=1)

# X
X_social_self_esteem = ['hexaco_4', 'hexaco_28', 'hexaco_52']
df['X_social_self_esteem'] = df[X_social_self_esteem].mean(axis=1)

X_social_boldness = ['hexaco_10', 'hexaco_34', 'hexaco_58']
df['X_social_boldness'] = df[X_social_boldness].mean(axis=1)

X_sociability = ['hexaco_16', 'hexaco_40']
df['X_sociability'] = df[X_sociability].mean(axis=1)

X_liveliness = ['hexaco_22', 'hexaco_46']
df['X_liveliness'] = df[X_liveliness].mean(axis=1)

# A
A_forgiveness = ['hexaco_3', 'hexaco_27']
df['A_forgiveness'] = df[A_forgiveness].mean(axis=1)

A_gentleness = ['hexaco_9', 'hexaco_33', 'hexaco_51']
df['A_gentleness'] = df[A_gentleness].mean(axis=1)

A_flexibility = ['hexaco_15', 'hexaco_39', 'hexaco_57']
df['A_flexibility'] = df[A_flexibility].mean(axis=1)

A_patience = ['hexaco_21', 'hexaco_45']
df['A_patience'] = df[A_patience].mean(axis=1)

# C
C_organization = ['hexaco_2', 'hexaco_26']
df['C_organization'] = df[C_organization].mean(axis=1)

C_diligence = ['hexaco_8', 'hexaco_33']
df['C_diligence'] = df[C_diligence].mean(axis=1)

C_perfectionism = ['hexaco_14', 'hexaco_38', 'hexaco_52']
df['C_perfectionism'] = df[C_perfectionism].mean(axis=1)

C_prudence = ['hexaco_20', 'hexaco_44', 'hexaco_56']
df['C_prudence'] = df[C_prudence].mean(axis=1)

# O
O_aesthetic_appreciation = ['hexaco_1', 'hexaco_25']
df['O_aesthetic_apreciation'] = df[O_aesthetic_appreciation].mean(axis=1)

O_inquisitiveness = ['hexaco_7', 'hexaco_31']
df['O_inquisitiveness'] = df[O_inquisitiveness].mean(axis=1)

O_creativity = ['hexaco_13', 'hexaco_37', 'hexaco_49']
df['O_creativity'] = df[O_creativity].mean(axis=1)

O_unconventionality = ['hexaco_19', 'hexaco_43', 'hexaco_55']
df['O_unconventionality'] = df[O_unconventionality].mean(axis=1)

# Box plots


def box(dataframe, num_rows, num_cols, group, x, *y):
    """
    Box plot of the quantitative data, hued by group
    :param dataframe: Pandas dataframe
    :param num_rows: number of rows
    :param num_cols: number of columns
    :param group: group to be hued by
    :param x: variable x
    :param y: variable y (can be multiple)
    :return: return all box plots in one image
    """
    fig, ax = plt.subplots(num_rows, num_cols)
    ax = ax.flatten()
    for i, axi in enumerate(ax):
        if i < len(y):
            sns.boxplot(x=x, y=y[i], hue=group, data=dataframe, ax=axi)
            axi.set_title("Box plot plot of {}".format(y[i]))
        else:
            fig.delaxes(axi)
    plt.tight_layout()
    return plt.show()


# print(box(df, 2, 3, None, None, 'H', 'E', 'X', 'A', 'C', 'O'))
# print(box(df, 1, 2, None, None, 'dti'))
# Removing outliers
df = df[df.H > 1.5]
df = df[df.E > 1.5]
df = df[df.A > 1.2]
df = df[df.A < 4.9]
df = df[df.C > 2]
df = df[df.O > 2.1]
df = df[df.dti > 28]
df = df[df.dti < 86]
# print(box(df, 2, 3, None, None, 'H', 'E', 'X', 'A', 'C', 'O'))

# Histograms


def histogram_multiple_df(num_cols, num_rows, *dataframes):
    """
    :param num_cols: number (integers) of columns for the plot
    :param num_rows: number (integers) of rows for the plot
    :param dataframes: dataframes or sepcific rows (ex. df['sex]
    :return: Histogram(s) plot(s)
    """
    fig, ax = plt.subplots(num_rows, num_cols)
    ax = ax.flatten()
    for i, axi in enumerate(ax):
        if i < len(dataframes):
            sns.histplot(dataframes[i], ax=axi, kde=True, color='crimson')
            axi.set_title("Histogram of {}".format(dataframes[i].name))
        else:
            fig.delaxes(axi)
    plt.tight_layout(h_pad=2)
    plt.subplots_adjust(wspace=2)
    return plt.show()


"""print(histogram_multiple_df(4, 2, df['H'], df['E'], df['X'],
                            df['A'], df['C'], df['O'], df['dti'], df['age']))"""
# Testing for normality


def normality(method, *variables):
    for arg in variables:
        print(pg.normality(data=df[arg], method=method, alpha=0.05))
    return '\n'


# print(normality('normaltest', "H", "E", "X", "A", "C", "O", "dti_all"))

# Shapiro-Wilk Can be used above by changing 'method'


def shapiro_wilk_test(*data):
    """
    Performs the Shapiro-Wilk test on multiple inputs.
    :param data: one or more arrays representing the data to test
    Returns: None
    """
    for i in data:
        stat, p = stats.shapiro(df[i])
        print('Statistic: {:.3f}'.format(stat))
        print('p-value: {:.3f}'.format(p))
        if p > 0.05:
            print('Data probably follows a Gaussian distribution. '
                  'For variable {}\n'.format(i))
        else:
            print('Data probably does not follow a Gaussian distribution. '
                  'For variable {}\n'.format(i))


# print(shapiro_wilk_test('H', 'E', 'X', 'A', 'C', 'O', 'dti_all', 'sex_num'))

# Q-Q plot


def qq_multiple_df(num_cols, num_rows, *dataframes):
    """
    :param num_cols: number (integers) of columns for the plot
    :param num_rows: number (integers) of rows for the plot
    :param dataframes: dataframes or sepcific rows (ex. df['sex]
    :return: Quantile-Quantile plot(s)
    """
    fig, ax = plt.subplots(num_rows, num_cols)
    ax = ax.flatten()
    for i, axi in enumerate(ax):
        if i < len(dataframes):
            pg.qqplot(dataframes[i], ax=axi)
            axi.set_title("Q-Q plot of {}".format(dataframes[i].name))
        else:
            fig.delaxes(axi)
    plt.tight_layout()
    return plt.show()


"""print(qq_multiple_df(3, 2, df['H'], df['E'], df['X'], df['A'], df['C'], df['O']))
print(qq_multiple_df(2, 2, df['dti'], df['profit_loss_thinking'],
                     df['dichotomous_belief'], df['preference_for_dichotomy']))"""


"""
print(qq_multiple_df(4, 3, df['H_sincerity'], df['H_fairness'], df['H_greed_avoidance'], df['H_modesty'],
                     df['E_fearfulness'], df['E_anxiety'], df['E_dependence'], df['E_sentimentality']))

print(qq_multiple_df(4, 3, df['X_social_self_esteem'], df['X_social_boldness'], df['X_sociability'], df['X_liveliness'],
                     df['A_forgiveness'], df['A_gentleness'], df['A_flexibility'], df['A_patience']))

print(qq_multiple_df(4, 3, df['C_organization'], df['C_diligence'], df['C_perfectionism'], df['C_prudence'],
                     df['O_aesthetic_apreciation'], df['O_inquisitiveness'], df['O_creativity'],
                     df['O_unconventionality']))
"""


def qqdti():
    """
    Quantile-quantile plot for DTI and
    :return: QQ-plot of DTI
    """
    fig, axes = plt.subplots(2, 2, figsize=(4, 4))
    fig.tight_layout(h_pad=2)
    pg.qqplot(df['dti'], dist='norm', ax=axes[0, 0],
              confidence=0.95).set_title('DTI', size=10)
    pg.qqplot(df['preference_for_dichotomy'], dist='norm', ax=axes[0, 1],
              confidence=0.95).set_title('Preference for Dichotomy', size=10)
    pg.qqplot(df['dichotomous_belief'], dist='norm', ax=axes[1, 0],
              confidence=0.95).set_title('Dichotomous Belief', size=10)
    pg.qqplot(df['profit_loss_thinking'], dist='norm', ax=axes[1, 1],
              confidence=0.95).set_title('Profit and Loss Thinking', size=10)
    return plt.show()


# Violin Plot


def violin(dataframe, num_rows, num_cols, group, x, *y):
    """
    Violin plot of the quantitative data, hued by group
    :param dataframe: Pandas dataframe
    :param num_rows: number of rows
    :param num_cols: number of columns
    :param group: group to be hued by
    :param x: variable x
    :param y: variable y (can be multiple)
    :return: return all box plots in one image
    """
    fig, ax = plt.subplots(num_rows, num_cols)
    ax = ax.flatten()
    for i, axi in enumerate(ax):
        if i < len(y):
            sns.violinplot(x=x, y=y[i], hue=group, data=dataframe, ax=axi)
            axi.set_title("Violin plot plot of {}".format(y[i]))
        else:
            fig.delaxes(axi)
    plt.tight_layout()
    return plt.show()


# print(violin(df, 2, 3, 'kinsey', 'sex', 'H', 'E', 'X', 'A', 'C', 'O'))

# Reliability


def reliability():
    """
    Prints the reliability values for HEXACO and DTI
    :return: None
    """
    print(f"Cronbach's alpha for H: {pg.cronbach_alpha(data=df[H])}")
    print(f"Cronbach's alpha for E: {pg.cronbach_alpha(data=df[E])}")
    print(f"Cronbach's alpha for X: {pg.cronbach_alpha(data=df[X])}")
    print(f"Cronbach's alpha for A: {pg.cronbach_alpha(data=df[A])}")
    print(f"Cronbach's alpha for C: {pg.cronbach_alpha(data=df[C])}")
    print(f"Cronbach's alpha for O: {pg.cronbach_alpha(data=df[O])}\n")

    print(f"Cronbach's alpha for Preference for Dichotomy:"
          f" {pg.cronbach_alpha(data=df[preference_for_dichotomy])}")
    print(f"Cronbach's alpha for Dichotomous Belief: "
          f"{pg.cronbach_alpha(data=df[dichotomous_belief])}")
    print(f"Cronbach's alpha for Profit and and Loss Thinking: "
          f"{pg.cronbach_alpha(data=df[profit_loss_thinking])}")
    print(f"Cronbach's alpha for total DTI: "
          f"{pg.cronbach_alpha(data=df[dti])}")
    return "\n"


# print(reliability())

def reliability_test(*variables):
    """
    Calculates the reliability for the variables of interest.
    :param variables: variables of interest in the Pandas dataframe
    :return: None
    """
    for var in variables:
        var = df[var]
        alpha = pg.cronbach_alpha(data=var)
        print(f"Reliability for variable {alpha}")
    return None


# print(reliability_test(H, E, X, A, C, O, dti))

# Residiual plots


def residplot_dti():
    """
    Returns a figure of HEXACO and DTI residual plots
    :return: residual plot
    """
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 9))
    sns.residplot(df, x='dti', y='H', ax=axes[0, 0])
    sns.residplot(df, x='dti', y='E', ax=axes[0, 1])
    sns.residplot(df, x='dti', y='X', ax=axes[0, 2])
    sns.residplot(df, x='dti', y='A', ax=axes[1, 0])
    sns.residplot(df, x='dti', y='C', ax=axes[1, 1])
    sns.residplot(df, x='dti', y='O', ax=axes[1, 2])
    return plt.show()


# print(residplot_dti())


def pearsonr_all():
    """
    Prints the correlation between HEXACO and DTI,
    :return: a correlation matrix.
    """
    print(f"{stats.pearsonr(x=df['dti'], y=df['H'])} for H")
    print(f"{stats.pearsonr(x=df['dti'], y=df['E'])} for E")
    print(f"{stats.pearsonr(x=df['dti'], y=df['X'])} for X")
    print(f"{stats.pearsonr(x=df['dti'], y=df['A'])} for A")
    print(f"{stats.pearsonr(x=df['dti'], y=df['C'])} for C")
    print(f"{stats.pearsonr(x=df['dti'], y=df['O'])} for O")

    df2 = df[['H', 'E', 'X', 'A', 'C', 'O', 'dti']]

    correlation_matrix = df2.corr(method='pearson').round(2)
    matrix_lower = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    sns.set(rc={'figure.figsize': (13, 10)})
    sns.heatmap(data=correlation_matrix, annot=True, mask=matrix_lower)
    return plt.show()


# print(pearsonr_all())


def regplot_dti():
    """
    Regression plots of the HEXACO and DTI values
    :return: Regression plots in one image
    """
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 9))
    sns.regplot(df, x='dti', y='H', scatter_kws={'color': 'blue'},
                line_kws={'color': 'red'}, ax=axes[0, 0])
    sns.regplot(df, x='dti', y='E', scatter_kws={'color': 'blue'},
                line_kws={'color': 'red'}, ax=axes[0, 1])
    sns.regplot(df, x='dti', y='X', scatter_kws={'color': 'blue'},
                line_kws={'color': 'red'}, ax=axes[0, 2])
    sns.regplot(df, x='dti', y='A', scatter_kws={'color': 'blue'},
                line_kws={'color': 'red'}, ax=axes[1, 0])
    sns.regplot(df, x='dti', y='C', scatter_kws={'color': 'blue'},
                line_kws={'color': 'red'}, ax=axes[1, 1])
    sns.regplot(df, x='dti', y='O', scatter_kws={'color': 'blue'},
                line_kws={'color': 'red'}, ax=axes[1, 2])
    return plt.show()


# print(regplot_dti())


# Multinomial Logistic Regression
model = sm.MNLogit.from_formula("course_num ~ H + E + X + A + C + O + dti + "
                                "sex + kinsey + sex:kinsey + age", data=df).fit()
result = model.summary()
print(result)



# Saving dfs
# df.to_csv('analysis.csv')
# df.to_excel('analysis.xlsx')

# Saving model summary
# results_as_html = result.tables[1].as_html()
# res = pd.read_html(results_as_html, header=0, index_col=0)[0]
# res.to_html('summary.html')
