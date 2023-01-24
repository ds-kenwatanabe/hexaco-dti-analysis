import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

"""
# Exemplo checha
hexaco_cols_reverse = ['sessE', 'oãs', 'solpmexe', 'od', 'esrever']

df = df.replace(hexaco_dict)
df[hexaco_cols_reverse] = 5 - df[hexaco_cols_reverse]
"""

df = pd.read_csv('pesquisa-mestrado-chris_January+23,+2023_10.23.csv')

# Dropping first two rows (import id)
df = df.drop(df.index[[0, 1]])

# Changing courses to major areas of knowledge
df.loc[df["course"] == "Outro", "course"] = df[df["course"] == "Outro"]["course2"]

course_names = df['course'].value_counts().index.tolist()
# print(course_names)

# Dropping course2 column and NaN values
df = df.drop(['course2'], axis=1)
df = df.dropna()
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

print(df['course'].describe())

# Changing course semester to numerical values
df['semester'] = df.semester.replace({'1º': 1, '2º': 2, '3º': 3, '4º': 4, '5º': 5,
                                      '6º': 6, '7º': 7, '8º': 8, '9º': 9,
                                      '10º': 10, '11º': 11, '12º': 12})

# Excluding sex other than male/female
df = df[df['sex'] != 'Outro (especifique)']
# Excluding asexuals
df = df[df['kinsey'] != 'Não tenho atração por nenhum gênero']

df['sexo'] = df.sex.replace({'Masculino': 1, 'Feminino': 2})
df['course_num'] = df.course.replace({'Ciências Humanas': 1,
                                  'Ciências Biológicas': 2,
                                  'Ciências Exatas': 3})
print(df['course'].describe())
print(df['sex'].describe())
# Total DTI
# Changing DTI values to numerical values
dti_dict = {'1 - Discordo Totalmente': 1, '2 - Discordo': 2,
            '3 - Discordo um Pouco': 3, '4 - Concordo um Pouco': 4,
            '5 - Concordo': 5, '6 - Concordo Totalmente': 6}

dti_all = ['dti_1', 'dti_4', 'dti_7', 'dti_10', 'dti_13',
             'dti_2', 'dti_5', 'dti_8', 'dti_11', 'dti_14',
             'dti_3', 'dti_6', 'dti_9', 'dti_12', 'dti_15']

df[dti_all] = df[dti_all].replace(dti_dict)
df['dti_all'] = df[dti_all].sum(axis=1)

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

# Changig Kinsey scale to heterosexual, bi and homo
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


"""
fig, ax = plt.subplots(2, 3)
sns.histplot(df, x='H', hue='sex', kde=True, ax=ax[0, 0])
sns.histplot(df, x='E', hue='sex', kde=True, ax=ax[0, 1])
sns.histplot(df, x='X', hue='sex', kde=True, ax=ax[0, 2])
sns.histplot(df, x='A', hue='sex', kde=True, ax=ax[1, 0])
sns.histplot(df, x='C', hue='sex', kde=True, ax=ax[1, 1])
sns.histplot(df, x='O', hue='sex', kde=True, ax=ax[1, 2])
plt.show()
"""

"""
# Não ficou bom
da = df[['H', 'E', 'X', 'A', 'C', 'O', 'sex', 'kinsey']]
variables = ['H', 'E', 'X', 'A', 'C', 'O']
pers = sns.PairGrid(da, vars=variables, hue='sex')
pers.map(sns.histplot)
# plt.show()
"""

# sns.residplot(df, x='dti_all', y='H')

variables = ['H', 'E', 'X', 'A', 'C', 'O']

def correlatios():
    for n in variables:
        return f"Correlations are: {pearsonr(x=df['dti_all'], y=variables)}"

print(pearsonr(x=df['dti_all'], y=df['H']))
print(pearsonr(x=df['dti_all'], y=df['E']))
print(pearsonr(x=df['dti_all'], y=df['X']))
print(pearsonr(x=df['dti_all'], y=df['A']))
print(pearsonr(x=df['dti_all'], y=df['C']))
print(pearsonr(x=df['dti_all'], y=df['O']))

df2 = df[['H', 'E', 'X', 'A', 'C', 'O', 'dti_all']]

correlation_matrix = df2.corr(method='pearson').round(2)

sns.set(rc={'figure.figsize':(11.7, 8.27)})
sns.heatmap(data=correlation_matrix, annot=True)
plt.show()

sns.regplot(df, x='dti_all', y='O')
# plt.show()
