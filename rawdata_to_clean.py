import pandas as pd
import numpy as np
import os


fr = pd.DataFrame()

in_directory = 'datas/'
out_directory = 'datas_clean/'

s_hetu = pd.Series([], index=pd.Index([], name='henkilotunnus'), name='id')

if os.path.isfile('data_clean.csv'):
    os.remove('data_clean.csv')

first_file = True
with open('data_clean.csv', 'a', encoding="utf-8") as fn_out:
    for fn in os.listdir(in_directory):
        fn_name, fn_ext = os.path.splitext(fn)
        if (fn_name[0] != '~') & (fn_ext.lower() == '.csv'):

            # Some columns have bad entries, the fixes below make the data valid
            year, month = fn_name[0:4], fn_name[4:6]

            fr_month = pd.read_csv(in_directory+fn, sep=';', encoding='windows-1252', dtype=str)
            fr_month.dropna(inplace=True, how='all')

            fr_month.loc[fr_month['tyottomalkamispvmnro'] == '0', 'tyottomalkamispvmnro'] = np.nan
            fr_month['tyottomalkamispvmnro'] = fr_month['tyottomalkamispvmnro'].str.replace(' ', '')

            bad1 = fr_month['vvkk'].str.len() == 3
            bad2 = fr_month['vvkk'].str.len() == 4
            fr_month.loc[bad1, 'vvkk'] = '200' + fr_month.loc[bad1, 'vvkk']
            fr_month.loc[bad2, 'vvkk'] = '20' + fr_month.loc[bad2, 'vvkk']
            fr_month['vvkk'] = fr_month['vvkk'] + '01'
            fr_month.rename(columns={'vvkk':'vvvvkk'}, inplace=True)

            if (year == '2012') and (month == '05'):
                fr_month['vvvvkk'] = '20120501'

            #fr_month['tyovoimatsto'] = fr_month['tyovoimatsto'].str.lstrip('0')
            fr_month['ammattikoodi'] = fr_month['ammattikoodi'].str.pad(5, fillchar='0')
            fr_month['kassa'] = fr_month['kassa'].str.pad(3, fillchar='0')
            fr_month['tyokokemuskoodi'] = fr_month['tyokokemuskoodi'].str.pad(2, fillchar='0')
            fr_month['koul_koulutuskoodi'] = fr_month['koul_koulutuskoodi'].str.pad(6, fillchar='0')
            fr_month['tyooleskelulupa'] = fr_month['tyooleskelulupa'].str.pad(2, fillchar='0')
            fr_month['kansalaisuus'] = fr_month['kansalaisuus'].str.pad(3, fillchar='0')
            fr_month['aidinkielikoodi'] = fr_month['aidinkielikoodi'].str.pad(2, fillchar='0')
            fr_month['voimolevatyollkoodi'] = fr_month['voimolevatyollkoodi'].str.pad(2, fillchar='0')
            fr_month['kuntakoodi'] = fr_month['kuntakoodi'].str.pad(3, fillchar='0')
            fr_month['tyovoimatsto'] = fr_month['tyovoimatsto'].str.pad(4, fillchar='0')
            fr_month['peruskoulutus'] = fr_month['peruskoulutus'].str.pad(2, fillchar='0')
            fr_month['tyovoimatstoyks'] = fr_month['tyovoimatstoyks'].str.pad(2, fillchar='0')
            fr_month['postinumero'] = fr_month['postinumero'].str.pad(5, fillchar='0')

            #if int(year) < 2012:
            #    fr_month['ammattikoodi'] = fr_month['ammattikoodi'].map(ammattikoodi2012)

            fr_month.loc[fr_month['voimolevatyollkoodi'] == 'zz', 'voimolevatyollkoodi'] = np.nan
            fr_month.loc[fr_month['tyooleskelulupa'] == 'zz', 'tyooleskelulupa'] = np.nan
            fr_month.loc[fr_month['peruskoulutus'] == 'zz', 'peruskoulutus'] = np.nan
            fr_month.loc[fr_month['ika'].astype(int) == 0, 'ika'] = np.nan
            fr_month.loc[fr_month['EGR'] == '#N/A', 'EGR'] = np.nan
            #fr_month.loc[fr_month['postinumero'].str.len() < 5, 'postinumero'] = np.nan

            # Add 'spv' column to denote birthday (used for calculating the age)
            to_spv = lambda s: '%s-%s-%s' % (str(1900 + int(s[4:6])) if int(s[4:6]) > 16 else str(2000 + int(s[4:6])), s[2:4], s[0:2])
            fr_month['spv'] = pd.to_datetime(fr_month['henkilotunnus'].apply(to_spv), format='%Y-%m-%d')

            # Drop 'henkilotunnus' that occurs more than once
            hetu_total = fr_month['henkilotunnus'].value_counts()
            invalid = hetu_total[hetu_total > 1].index.values
            rows_total = len(fr_month)
            fr_month.set_index('henkilotunnus', inplace=True)
            fr_month.drop(invalid, inplace=True)
            fr_month.reset_index(inplace=True)
            rows_valid = len(fr_month)

            # Update 'henkilotunnus' => id
            cur_hetu = fr_month['henkilotunnus'].values
            old_hetu = set(s_hetu.index.values)
            new_hetu = [hetu for hetu in cur_hetu if hetu not in old_hetu]
            new_id = np.arange(len(old_hetu), len(old_hetu) + len(new_hetu))
            s_add = pd.Series(new_id, index=pd.Index(new_hetu, name='henkilotunnus'), name='id')
            s_hetu = pd.concat([s_hetu, s_add])

            # Replace 'henkilotunnus' => id in data set
            fr_month['henkilotunnus'] = fr_month['henkilotunnus'].map(s_hetu)

            print("Filename: %s" % fn, "(valid rows %d/%d)" % (rows_valid, rows_total))
            print("\tTotal hetu(s): %d+=%d" % (len(old_hetu), len(new_hetu)))

            fr_month = fr_month\
            [['henkilotunnus','ammattikoodi','kassa','tyokokemuskoodi','koul_koulutuskoodi',
             'tyottomalkamispvmnro','tyooleskelulupa','kansalaisuus','ika','ammattinimike',
             'aidinkielikoodi','voimolevatyollkoodi','kuntakoodi','tyovoimatsto',
             'peruskoulutus','postinumero','tyovoimatstoyks','supu','EGR','spv', 'vvvvkk','kerayspv']]

            if (year == '2011') and (month == '11'):
                pass
            else:
                print(in_directory+fn, '=>', 'data_clean.csv')
                fr_month.to_csv(fn_out, sep=';', encoding='utf-8', index=None, header=first_file)
                first_file = False



