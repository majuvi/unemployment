# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os

# Data processing
'''
henkilotunnus                               => henkilotunnus (171258)
ammattikoodi                                => ammattikoodi (1111), ammattikoodi_1 (11), ammattikoodi_2 (51), ammattikoodi_3 (137), ammattikoodi_4 (420)
tyokokemuskoodi                             => tyokokemuskoodi (3)
koulutuskoodi                               => koulutuskoodi (1599), koulutusastekoodi_1 (9), koulutusastekoodi_2 (16), koulutusalakoodi_1 (12), koulutusalakoodi_2 (32), koulutusalakoodi_3 (99)
ika                                         => ika (47)
supukoodi                                   => supukoodi (2)
aidinkielikoodi                             => aidinkielikoodi (124)
kansalaisuuskoodi                           => kansalaisuuskoodi (159)
kuntakoodi                                  => 
vvvvkk                                      => vvvvkk (106)
voimolevatyollkoodi                         => voimolevatyollkoodi (7), voimolevatyollkoodi_prev (8)
EGR                         -------------------------------------------------------------------------------------------- (irrelevantti?)
tyooleskelulupa             -------------------------------------------------------------------------------------------- (irrelevantti?)
tyooleskelulupa             -------------------------------------------------------------------------------------------- (muuttunut koodisto)
postinumero                 -------------------------------------------------------------------------------------------- (keräys 2014 ->)
kassa                       -------------------------------------------------------------------------------------------- (liian monta NaN ?)
ammattinimike               -------------------------------------------------------------------------------------------- (liian vapaamuotoinen)
peruskoulutus               -------------------------------------------------------------------------------------------- (ks. koul_koulutuskoodi)
tyovoimatsto                -------------------------------------------------------------------------------------------- (irrelevantti?)
tyottomalkamispvmnro        -------------------------------------------------------------------------------------------- (puuttuu, resetoituu)
kerayspv                    -------------------------------------------------------------------------------------------- (puuttuu 2012 - 2014)
                                            => unemployed (2), other (2), unemployed_prev (2), other_prev (2)
'''
# Data correspondence table
'''
Data correspondence table
data.csv (koodi)                            data_koodit.csv (koodi -> arvo)
fr['henkilotunnus']
fr['kansalaisuuskoodi']                     # 3 digit string (000, ...., 999)
fr['aidinkielikoodi']                       # 2 digit string (aa, ..., zz)
fr['supukoodi']                             # 1 digit string (M, N)
fr['tyokokemuskoodi']                       # 2 digit string (00, 01, 02)
fr['ika']                                   # 2 digit string (18, 19, ..., 64)
fr['koulutuskoodi']                         # 6 digit string (001101, ..., 999999)
fr['koulutusastekoodi_1']                   # 1 digit string (0, ..., 9)
fr['koulutusastekoodi_2']                   # 2 digit string (02, ..., 91)
fr['koulutusalakoodi_1']                    # 2 digit string (00, ..., 10, 99)
fr['koulutusalakoodi_2']                    # 3 digit string (001, ..., 104, 999)
fr['koulutusalakoodi_3']                    # 4 digit string (0011, ...., 1048, 9999)
fr['ammattikoodi']                          # 5 digit string (01100, ..., XXXXX)
fr['ammattikoodi_1']                        # 1 digit string (0, ..., X)
fr['ammattikoodi_2']                        # 2 digit string (01, ..., XX)
fr['ammattikoodi_3']                        # 3 digit string (011, ..., XXX)
fr['ammattikoodi_4']                        # 4 digit string (0110, ..., XXXX)
fr['vvvvkk']                                # 7 digit string (2008-01, ...., 2017-12)
fr['voimolevatyollkoodi']                   # 2 digit string (00, 01, 02, 03, 04, 05, 06, 09)
fr['voimolevatyollkoodi_prev']              # 2 digit string (00, 01, 02, 03, 04, 05, 06, 09, 10)
'''
print()

import sys

columns_use = ['henkilotunnus', 'ammattikoodi', 'tyokokemuskoodi', 'koul_koulutuskoodi', 'ika', 'supu',
               'aidinkielikoodi', 'kansalaisuus', 'vvvvkk', 'voimolevatyollkoodi', 'spv']
columns_rename = columns={'kansalaisuus': 'kansalaisuuskoodi', 'koul_koulutuskoodi': 'koulutuskoodi', 'supu': 'supukoodi'}

# Read data set
fr = pd.read_csv('data_clean.csv', sep=';', encoding='utf-8', dtype=str, usecols=columns_use)
fr.rename(columns=columns_rename, inplace=True)
fr['vvvvkk'] = pd.to_datetime(fr['vvvvkk'])
fr['spv'] = pd.to_datetime(fr['spv'])

print("Features:")
print(fr.columns)
print("Total observations: %d (%d samples)" % (len(fr), len(fr['henkilotunnus'].unique())))
print(fr['vvvvkk'].min(), "---", fr['vvvvkk'].max())

# Valid dates
start, end = '2013-01-01', '2017-12-01'
all_dates = pd.to_datetime(pd.period_range(start, end, freq='M').to_timestamp())
obs_dates = pd.to_datetime(fr['vvvvkk'].unique()).sort_values()
missing_dates = pd.to_datetime([day for day in all_dates if day not in obs_dates]).sort_values()
print('All months:')
print(all_dates)
print('Obsevation months:')
print(obs_dates)
print('Missing months:')
print(missing_dates)

# Add 09 for 'not in data' and 10 for 'not observable'
print("Adding 09/10 status...")
fr['voimolevatyollkoodi'] = fr['voimolevatyollkoodi'].fillna(method='ffill').fillna(method='bfill')
vvvvkk = pd.Index(all_dates, name='vvvvkk')
fr = fr.groupby('henkilotunnus').apply(lambda fr: fr.set_index('vvvvkk').reindex(vvvvkk))
fr = fr.drop('henkilotunnus', axis=1)
fr = fr.reset_index()
fr.loc[fr['vvvvkk'].isin(obs_dates) & fr['voimolevatyollkoodi'].isnull(), 'voimolevatyollkoodi'] = '09'
fr.loc[fr['vvvvkk'].isin(missing_dates) & fr['voimolevatyollkoodi'].isnull(), 'voimolevatyollkoodi'] = '10'
print("Missing status:", fr['voimolevatyollkoodi'].isnull().sum())

print("Filling forward & backward...")
# Add covariates for observations with 'not in data' status by filling forward and backward
fr = fr.groupby('henkilotunnus').apply(lambda fr: fr.fillna(method='ffill').fillna(method='bfill'))

print("Adding real age...")
kerayspv = (fr['vvvvkk'] + pd.tseries.offsets.MonthEnd(1))
fr['ika'] = np.floor((kerayspv - fr['spv']) / np.timedelta64(1, 'Y'))
fr['ika'] = fr['ika'].astype(int)

# Censor not working age (very slight difference in covariate 'ika' and this 'ika' due to 'kerayspv' not last of month)
print("Setting censored status for not working age or retired...")
fr.loc[(fr['ika'] < 18) | (fr['ika'] > 64), 'voimolevatyollkoodi'] = '10'
# Replace every employment status with censored from the first retired observation
def censor06(s):
    retired = (s == '06')
    if retired.any():
        idx = np.argmax(retired.values)
        s.iloc[idx:] = '10'
    return s
fr['voimolevatyollkoodi'] = fr.groupby('henkilotunnus')['voimolevatyollkoodi'].apply(censor06)
# Remove samples with only retired / employed status
print("Dropping those with only censored, retired or employed status")
only_retired = fr.groupby('henkilotunnus')['voimolevatyollkoodi'].apply(lambda s: ((s == '10') | (s == '09')).all())
invalid_ids = only_retired[only_retired].index.values
invalid_obs = fr['henkilotunnus'].isin(invalid_ids)
fr = fr[~invalid_obs]
print("(%d/%d total)..." % (invalid_obs.sum(), len(invalid_ids)))

# Mapping old codes to new codes (PAL -> ISCO) (Convert ammattikoodi < 2014-06-01 to new ammattikoodi)
print("Converting old codes to new codes...")
fr.loc[fr['ammattikoodi'].str.contains('X') & (fr['ammattikoodi'].str[0] != 'X'), 'ammattikoodi'] = 'XXXXX'
ammatti_map = pd.read_csv('ammattikoodi_map2.csv', sep=';', encoding='windows-1252')
ammatti_map['PAL-ammattikoodi'] = ammatti_map['PAL-ammattikoodi'].str.pad(5,fillchar='0')
ammatti_map['ISCO-ammattikoodi'] = ammatti_map['ISCO-ammattikoodi'].astype(str).str.pad(5,fillchar='0')
ammatti_map = ammatti_map.set_index('PAL-ammattikoodi')['ISCO-ammattikoodi']
for koodi in fr['ammattikoodi'].unique():
    if koodi not in ammatti_map:
        ammatti_map[koodi] = koodi
convert_idx = fr['vvvvkk'] < '2014-06-01'
fr.loc[convert_idx, 'ammattikoodi'] = fr.loc[convert_idx, 'ammattikoodi'].map(ammatti_map)

print("Total observations: %d (%d samples)" % (len(fr), len(fr['henkilotunnus'].unique())))

print("Sorting...")
fr.sort_values(['henkilotunnus', 'vvvvkk'], inplace=True)

# Given boolean flags for missing codes, labels and unknown labels (with possibly excluded years) print observation and sample counts
def print_nan(koodi_nan=None, leima_nan=None, leima_zzz=None, missing_yrs = False):
    if not koodi_nan is None:
        print("koodi NaN: %d observations (%d samples)" % ((~missing_yrs & koodi_nan).sum(), len(fr.loc[~missing_yrs & koodi_nan, 'henkilotunnus'].unique())))
    if not leima_nan is None:
        print("leima NaN: %d observations (%d samples)" % ((~missing_yrs & leima_nan).sum(), len(fr.loc[~missing_yrs & leima_nan, 'henkilotunnus'].unique())))
    if not leima_zzz is None:
        print("leima ZZZ: %d observations (%d samples)" % ((~missing_yrs & leima_zzz).sum(), len(fr.loc[~missing_yrs & leima_zzz, 'henkilotunnus'].unique())))

print("=== voimolevatyollkoodi ===")
# Categorical employment status codes in data
tyollkoodit = {'00': 'Työllistetty',
               '01': 'Työssä yleisillä työmarkk.',
               '02': 'Työtön',
               '03': 'Lomautettu',
               '04': 'Lyhennetyllä työvkolla',
               '05': 'Työvoiman ulkopuolella',
               '06': 'Työttömyyseläkkeellä',
               '07': 'Työllistym. ed. palvelussa',
               '08': 'Koulutuksessa',
               '09': 'Ei havaittu', # Työllistynyt?
               '10': 'Ei havaittavissa'}
# Replace 2013 onwards to old, less informative, codes
fr['voimolevatyollkoodi'] = fr['voimolevatyollkoodi'].map({'00': '00', '01': '01', '02': '02', '03': '03', '04': '04',
                                                           '05': '05', '06': '06', '07': '05', '08': '05', '09': '09',
                                                           '10': '10'})
# Add labels
fr['voimolevatyoll'] = fr['voimolevatyollkoodi'].map(tyollkoodit)
koodi_nan, leima_nan = fr['voimolevatyollkoodi'].isnull(), fr['voimolevatyoll'].isnull()
print_nan(koodi_nan, leima_nan)
# Fill with 'Työtön'
fr.loc[koodi_nan | leima_nan, 'voimolevatyollkoodi'] = '02'
#fr['voimolevatyoll'] = fr['voimolevatyollkoodi'].map(tyollkoodit)
fr.drop(columns='voimolevatyoll', inplace=True)

print("=== kansalaisuuskoodi (excluding 2008-01 & 2008-02) ===")
# Categorical citizenship codes in data
kansakoodit = pd.read_csv('kansalaisuus_map.csv', encoding='utf-8', sep=';')
kansakoodit['Koodi'] = kansakoodit['Koodi'].astype(str).str.pad(3,fillchar='0')
kansakoodit = kansakoodit.set_index('Koodi')['Kansalaisuus']
kansakoodit = kansakoodit.str.lower()
kansakoodit.sort_index(inplace=True)
# Replace invalid values with unknown
fr.loc[(fr['kansalaisuuskoodi'] == 'zzz') | (fr['kansalaisuuskoodi'] == '000'), 'kansalaisuuskoodi'] = '999'
# Add labels
fr['kansalaisuus'] = fr['kansalaisuuskoodi'].map(kansakoodit)
#missing_yrs = ((fr['vvvvkk'] >= '2008-01') & (fr['vvvvkk'] <= '2008-02'))
koodi_nan, leima_nan, leima_zzz = fr['kansalaisuuskoodi'].isnull(), fr['kansalaisuus'].isnull(), fr['kansalaisuus'] == 'tuntematon'
print_nan(koodi_nan, leima_nan, leima_zzz) #print_nan(koodi_nan, leima_nan, leima_zzz, missing_yrs)
# Fill with 'tuntematon'
fr.loc[(koodi_nan | leima_nan | leima_zzz), 'kansalaisuuskoodi'] = '999' #fr.loc[(koodi_nan | leima_nan | leima_zzz) & ~missing_yrs, 'kansalaisuuskoodi'] = '999'
#fr['kansalaisuus'] = fr['kansalaisuuskoodi'].map(kansakoodit)
fr.drop(columns='kansalaisuus', inplace=True)

print("=== aidinkielikoodi ===")
# Categorical language codes in data
kielikoodit = pd.read_csv('aidinkielikoodi_map.csv', encoding='utf-8', sep=';', index_col=None, na_values=None, keep_default_na=False)
kielikoodit['Koodi'] = kielikoodit['Koodi'].str.lower()
kielikoodit = kielikoodit.set_index('Koodi')['Kieli']
kielikoodit = kielikoodit.str.lower()
kielikoodit.sort_index(inplace=True)
fr['aidinkielikoodi'] = fr['aidinkielikoodi'].str.lower()
# Replace invalid values with unknown
fr.loc[(fr['aidinkielikoodi'] == 'yy') | (fr['aidinkielikoodi'] == ''), 'aidinkielikoodi'] = 'zz'
# Add labels
fr['aidinkieli'] = fr['aidinkielikoodi'].map(kielikoodit)
koodi_nan, leima_nan, leima_zzz = fr['aidinkielikoodi'].isnull(), fr['aidinkieli'].isnull(), fr['aidinkieli'] == 'tuntematon'
print_nan(koodi_nan, leima_nan, leima_zzz)
# Fill with 'tuntematon'
fr.loc[koodi_nan | leima_nan | leima_zzz, 'aidinkielikoodi'] = 'zz'
#fr['aidinkieli'] = fr['aidinkielikoodi'].map(kielikoodit)
fr.drop(columns='aidinkieli', inplace=True)

print("=== tyokokemuskoodi ===")
# Categorical employment experience codes in data
kokemuskoodit = {'00': 'ei työkokemusta',
                 '01': 'vähän työkokemusta',
                 '02': 'riittävästi työkokemusta'}
# Add labels
fr['tyokokemus'] = fr['tyokokemuskoodi'].map(kokemuskoodit)
koodi_nan, leima_nan = fr['tyokokemuskoodi'].isnull(), fr['tyokokemus'].isnull()
print_nan(koodi_nan, leima_nan)
# Fill with 'riittävästi työkokemusta'
fr.loc[koodi_nan | leima_nan, 'tyokokemuskoodi'] = '02'
#fr['tyokokemus'] = fr['tyokokemuskoodi'].map(kokemuskoodit)
fr.drop(columns='tyokokemus', inplace=True)

print("=== supukoodi ===")
# Categorical gender codes in data
supukoodi = {'M': 'Mies', 'N': 'Nainen'}
# Add labels
fr['supu'] = fr['supukoodi'].map(supukoodi)
koodi_nan, leima_nan = fr['supukoodi'].isnull(), fr['supu'].isnull()
print_nan(koodi_nan, leima_nan)
fr.drop(columns='supu', inplace=True)

print("=== ika ===")
koodi_nan = fr['ika'].isnull()
print_nan(koodi_nan)

print("=== koulutuskoodi ===")
# Categorical education codes in data
koul = pd.read_csv('koulutuskoodi_map.csv', encoding='utf-8', sep=';')
koul['Koulutuskoodi'] = koul['Koulutuskoodi'].astype(str).str.pad(6,fillchar='0')
koul = koul.set_index('Koulutuskoodi')
koul['Koulutusaste, taso 1'] = koul['Koulutusaste, taso 1'].astype(str).str.pad(1, fillchar='0')
koul['Koulutusaste, taso 2'] = koul['Koulutusaste, taso 2'].astype(str).str.pad(2, fillchar='0')
koul['Koulutusala, taso 1'] = koul['Koulutusala, taso 1'].astype(str).str.pad(2, fillchar='0')
koul['Koulutusala, taso 2'] = koul['Koulutusala, taso 2'].astype(str).str.pad(3, fillchar='0')
koul['Koulutusala, taso 3'] = koul['Koulutusala, taso 3'].astype(str).str.pad(4, fillchar='0')
koulaste1 = koul.set_index(['Koulutusaste, taso 1'])['Koulutusaste, taso 1, suomi'].drop_duplicates().sort_index()
koulaste2 = koul.set_index(['Koulutusaste, taso 2'])['Koulutusaste, taso 2, suomi'].drop_duplicates().sort_index()
koulala1 = koul.set_index(['Koulutusala, taso 1'])['Koulutusala, taso 1, suomi'].drop_duplicates().sort_index()
koulala2 = koul.set_index(['Koulutusala, taso 2'])['Koulutusala, taso 2, suomi'].drop_duplicates().sort_index()
koulala3 = koul.set_index(['Koulutusala, taso 3'])['Koulutusala, taso 3, suomi'].drop_duplicates().sort_index()
# Add Koulutusaste (2) & Koulutusala (3) codes
fr['koulutusastekoodi_1'] = fr['koulutuskoodi'].map(koul['Koulutusaste, taso 1'])
fr['koulutusastekoodi_2'] = fr['koulutuskoodi'].map(koul['Koulutusaste, taso 2'])
fr['koulutusalakoodi_1'] = fr['koulutuskoodi'].map(koul['Koulutusala, taso 1'])
fr['koulutusalakoodi_2'] = fr['koulutuskoodi'].map(koul['Koulutusala, taso 2'])
fr['koulutusalakoodi_3'] = fr['koulutuskoodi'].map(koul['Koulutusala, taso 3'])
# Add Koulutusaste (2) & Koulutusala (3) labels
fr['koulutusaste_1'] = fr['koulutuskoodi'].map(koul['Koulutusaste, taso 1, suomi'])
fr['koulutusaste_2'] = fr['koulutuskoodi'].map(koul['Koulutusaste, taso 2, suomi'])
fr['koulutusala_1'] = fr['koulutuskoodi'].map(koul['Koulutusala, taso 1, suomi'])
fr['koulutusala_2'] = fr['koulutuskoodi'].map(koul['Koulutusala, taso 2, suomi'])
fr['koulutusala_3'] = fr['koulutuskoodi'].map(koul['Koulutusala, taso 3, suomi'])
# NaNs
koodi_nan = fr['koulutuskoodi'].isnull() | fr['koulutusastekoodi_1'].isnull() | fr['koulutusastekoodi_2'].isnull() | \
            fr['koulutusalakoodi_1'].isnull() | fr['koulutusalakoodi_2'].isnull() | fr['koulutusalakoodi_3'].isnull()
leima_nan = (fr['koulutusaste_1'].isnull() | fr['koulutusaste_2'].isnull() | \
             fr['koulutusala_1'].isnull() | fr['koulutusala_2'].isnull() | fr['koulutusala_3'].isnull())
leima_zzz = (fr['koulutuskoodi'] == '999999')
print_nan(koodi_nan, leima_nan, leima_zzz)
# Fill with 'tuntematon'
fr.loc[koodi_nan | leima_nan | leima_zzz, 'koulutuskoodi'] = '999999'
fr['koulutusastekoodi_1'] = fr['koulutuskoodi'].map(koul['Koulutusaste, taso 1'])
fr['koulutusastekoodi_2'] = fr['koulutuskoodi'].map(koul['Koulutusaste, taso 2'])
fr['koulutusalakoodi_1'] = fr['koulutuskoodi'].map(koul['Koulutusala, taso 1'])
fr['koulutusalakoodi_2'] = fr['koulutuskoodi'].map(koul['Koulutusala, taso 2'])
fr['koulutusalakoodi_3'] = fr['koulutuskoodi'].map(koul['Koulutusala, taso 3'])
#fr['koulutusaste_1'] = fr['koulutuskoodi'].map(koul['Koulutusaste, taso 1, suomi'])
#fr['koulutusaste_2'] = fr['koulutuskoodi'].map(koul['Koulutusaste, taso 2, suomi'])
#fr['koulutusala_1'] = fr['koulutuskoodi'].map(koul['Koulutusala, taso 1, suomi'])
#fr['koulutusala_2'] = fr['koulutuskoodi'].map(koul['Koulutusala, taso 2, suomi'])
#fr['koulutusala_3'] = fr['koulutuskoodi'].map(koul['Koulutusala, taso 3, suomi'])
fr.drop(columns=['koulutusaste_1', 'koulutusaste_2', 'koulutusala_1', 'koulutusala_2', 'koulutusala_3'],
        inplace=True)

print("=== Ammattikoodi ===")
# Categorical job codes in data
ammattiluokitus = pd.read_csv('ammattiluokitus2017TEM.csv', sep=';', encoding='utf-8', dtype=str)
ammattitaso1 = ammattiluokitus[ammattiluokitus['koodi'].str.len() == 1].set_index('koodi')['nimike']
ammattitaso2 = ammattiluokitus[ammattiluokitus['koodi'].str.len() == 2].set_index('koodi')['nimike']
ammattitaso3 = ammattiluokitus[ammattiluokitus['koodi'].str.len() == 3].set_index('koodi')['nimike']
ammattitaso4 = ammattiluokitus[ammattiluokitus['koodi'].str.len() == 4].set_index('koodi')['nimike']
# Add ammattikoodi codes (4) and ammatti labels (4)
fr['ammattikoodi_1'] = fr['ammattikoodi'].str[:1]
fr['ammattikoodi_2'] = fr['ammattikoodi'].str[:2]
fr['ammattikoodi_3'] = fr['ammattikoodi'].str[:3]
fr['ammattikoodi_4'] = fr['ammattikoodi'].str[:4]
fr['ammatti_1'] = fr['ammattikoodi_1'].map(ammattitaso1)
fr['ammatti_2'] = fr['ammattikoodi_2'].map(ammattitaso2)
fr['ammatti_3'] = fr['ammattikoodi_3'].map(ammattitaso3)
fr['ammatti_4'] = fr['ammattikoodi_4'].map(ammattitaso4)
# NaNs
koodi_nan = fr['ammattikoodi'].isnull() | fr['ammattikoodi_1'].isnull() |fr['ammattikoodi_2'].isnull() \
            | fr['ammattikoodi_3'].isnull() | fr['ammattikoodi_4'].isnull()
leima_nan = (fr['ammatti_1'].isnull() | fr['ammatti_2'].isnull() | fr['ammatti_3'].isnull() | fr['ammatti_4'].isnull())
leima_zzz = (fr['ammattikoodi'] == 'XXXXX')
print_nan(koodi_nan, leima_nan, leima_zzz)
# Fill with 'tuntematon'
fr.loc[koodi_nan | leima_nan | leima_zzz, 'ammattikoodi'] = 'XXXXX'
fr['ammattikoodi_1'] = fr['ammattikoodi'].str[:1]
fr['ammattikoodi_2'] = fr['ammattikoodi'].str[:2]
fr['ammattikoodi_3'] = fr['ammattikoodi'].str[:3]
fr['ammattikoodi_4'] = fr['ammattikoodi'].str[:4]
#fr['ammatti_1'] = fr['ammattikoodi_1'].map(ammattitaso1)
#fr['ammatti_2'] = fr['ammattikoodi_2'].map(ammattitaso2)
#fr['ammatti_3'] = fr['ammattikoodi_3'].map(ammattitaso3)
#fr['ammatti_4'] = fr['ammattikoodi_4'].map(ammattitaso4)
fr.drop(columns=['ammatti_1', 'ammatti_2', 'ammatti_3', 'ammatti_4'], inplace=True)

print("=== NaNs ===")
print(fr.isnull().sum())
# Boolean flag that stores any NaN observation
any_nan = (fr['kansalaisuuskoodi'] == '999') | (fr['aidinkielikoodi'] == 'zz') | (fr['koulutuskoodi'] == '999999') | (fr['ammattikoodi'] == 'XXXXX')
print("any NaN/unknown: %d observations (%d samples)" % (any_nan.sum(), len(fr.loc[any_nan, 'henkilotunnus'].unique())))

# Total code counts
# print fr['voimolevatyoll'].value_counts().sort_values(ascending=False)
# print fr['kansalaisuus'].value_counts().sort_values(ascending=False)
# print fr['aidinkieli'].value_counts().sort_values(ascending=False)
# print fr['tyokokemus'].value_counts().sort_values(ascending=False)
# print fr['supu'].value_counts().sort_values(ascending=False)
# print fr['ika'].value_counts().sort_index()
# print "koulutuskoodi:", len(koulaste1), len(koulaste2), len(koulala1), len(koulala2), len(koulala3)
# print fr['koulutusaste_1'].value_counts().sort_values(ascending=False)
# print fr['koulutusaste_2'].value_counts().sort_values(ascending=False)
# print fr['koulutusala_1'].value_counts().sort_values(ascending=False)
# print fr['koulutusala_2'].value_counts().sort_values(ascending=False)
# print fr['koulutusala_3'].value_counts().sort_values(ascending=False)
# print "ammattikoodi:", len(ammattitaso1), len(ammattitaso2), len(ammattitaso3), len(ammattitaso4)
# print fr['ammatti_1'].value_counts().sort_values(ascending=False)
# print fr['ammatti_2'].value_counts().sort_values(ascending=False)
# print fr['ammatti_3'].value_counts().sort_values(ascending=False)
# print fr['ammatti_4'].value_counts().sort_values(ascending=False)
# print fr['kokeilu'].value_counts().sort_values(ascending=False)

print("===== PROCESS STATE AND TRANSITION INFORMATION =====")
# Add state transitions
fr['voimolevatyollkoodi_prev'] = fr.groupby('henkilotunnus')['voimolevatyollkoodi'].apply(lambda s: s.shift(1).fillna('10'))
# Current & previous month's ending vvvvkk status
fr['unemployed'] = ((fr['voimolevatyollkoodi'] == '02') | (fr['voimolevatyollkoodi'] == '03')).astype(int)
fr['other'] = (~fr['unemployed'] & (fr['voimolevatyollkoodi'] != '10')).astype(int)
fr['unemployed_prev'] = ((fr['voimolevatyollkoodi_prev'] == '02') | (fr['voimolevatyollkoodi_prev'] == '03')).astype(int)
fr['other_prev'] = (~fr['unemployed_prev'] & (fr['voimolevatyollkoodi_prev'] != '10')).astype(int)

print("Saving dataset...")
# Remove observations with missing status
fr = fr[fr['voimolevatyollkoodi'] != '10']
# Write only codes to save space
write_cols = ['henkilotunnus', 'kansalaisuuskoodi', 'aidinkielikoodi', 'supukoodi', 'tyokokemuskoodi', 'ika',
          'koulutuskoodi', 'koulutusastekoodi_1', 'koulutusastekoodi_2', 'koulutusalakoodi_1', 'koulutusalakoodi_2', 'koulutusalakoodi_3',
          'ammattikoodi', 'ammattikoodi_1', 'ammattikoodi_2', 'ammattikoodi_3', 'ammattikoodi_4',
          'vvvvkk', 'voimolevatyollkoodi', 'voimolevatyollkoodi_prev', 'unemployed', 'other', 'unemployed_prev', 'other_prev']
fr.to_csv('data.csv', encoding='utf-8', sep=';', index=False, columns=write_cols)

print("Saving codes...")
kokemuskoodit = pd.Series(kokemuskoodit)
tyollkoodit = pd.Series(tyollkoodit)
supukoodi = pd.Series(supukoodi)
fr_code = pd.concat([
    pd.DataFrame({'column': 'kansalaisuuskoodi', 'key': kansakoodit.index, 'value': kansakoodit}),
    pd.DataFrame({'column': 'aidinkielikoodi', 'key': kielikoodit.index, 'value': kielikoodit}),
    pd.DataFrame({'column': 'supukoodi', 'key': supukoodi.index, 'value': supukoodi}),
    pd.DataFrame({'column': 'tyokokemuskoodi', 'key': kokemuskoodit.index, 'value': kokemuskoodit}),
    pd.DataFrame({'column': 'koulutusastekoodi_1', 'key': koulaste1.index, 'value': koulaste1}),
    pd.DataFrame({'column': 'koulutusastekoodi_2', 'key': koulaste2.index, 'value': koulaste2}),
    pd.DataFrame({'column': 'koulutusalakoodi_1', 'key': koulala1.index, 'value': koulala1}),
    pd.DataFrame({'column': 'koulutusalakoodi_2', 'key': koulala2.index, 'value': koulala2}),
    pd.DataFrame({'column': 'koulutusalakoodi_3', 'key': koulala3.index, 'value': koulala3}),
    pd.DataFrame({'column': 'ammattikoodi_1', 'key': ammattitaso1.index, 'value': ammattitaso1}),
    pd.DataFrame({'column': 'ammattikoodi_2', 'key': ammattitaso2.index, 'value': ammattitaso2}),
    pd.DataFrame({'column': 'ammattikoodi_3', 'key': ammattitaso3.index, 'value': ammattitaso3}),
    pd.DataFrame({'column': 'ammattikoodi_4', 'key': ammattitaso4.index, 'value': ammattitaso4}),
    pd.DataFrame({'column': 'voimolevatyollkoodi', 'key': tyollkoodit.index, 'value': tyollkoodit})
    ], ignore_index=True)
fr_code.to_csv('data_koodit.csv', encoding='utf-8', sep=';', index=False)


