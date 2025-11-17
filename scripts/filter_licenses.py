import pandas as pd

df = pd.read_csv('third_party_licenses.csv')
df = df[df['License'].str.contains('GPL|AGPL|LGPL|MPL', na=False)]

# if the license has a long agreement, only capture the title and skip the rest
df['License'] = df['License'].apply(lambda x: x.split('\n')[0])

df = df[~df['Name'].str.contains('quadprog')] # ignore quadprog
df.to_markdown('THIRD_PARTY_LICENSES.md', index=False)
