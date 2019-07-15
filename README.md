# Dota 2 Toxicity Analysis

```python
import pandas as pd

# Load in all the data

chat_df = pd.read_csv("chat.csv")
match_df = pd.read_csv("match.csv")
match_df.set_index('match_id', inplace=True)

# Using a bag of words model to rank each phrase

with open('positive_words.txt') as pos_words_file:
    positive_words = set(pos_words_file.read().split('\n')[:-1])
    
with open('negative_words.txt') as neg_words_file:
    negative_words = set(neg_words_file.read().split('\n')[:-1])

```


```python
# Analyse sentences and retrieve scores

def score_chat(row):
    score = 0
    for word in str(row['key']).split(" "):
        if word in positive_words:
            score += 1
        if word in negative_words:
            score -= 1
    return score

scores = chat_df.apply(score_chat, axis=1)
chat_df['score'] = scores
chat_df.drop(['key', 'slot', 'time', 'unit'], axis=1, inplace=True)
chat_df.to_csv('processed_chat.csv')

```


```python
score_df = pd.read_csv('processed_chat.csv')
```


```python
# Load clusters into score dataframe

def region_from_match_id(row):
    return match_df.iloc[row['match_id']]['cluster']

score_df['cluster'] = score_df.apply(region_from_match_id, axis=1)
score_df.to_csv('processed_chat.csv')

```


```python
from collections import defaultdict
import json

# Tally scores based on region

with open('region.json') as regions_file:
    regions = json.load(regions_file)

tally = defaultdict(int)

for region_code, region in regions.items():
    region_chats = score_df[score_df['cluster'] == int(region_code)]
    region_score = region_chats['score'].sum()
    tally[region] += region_score

```


```python
# Plot data on map

print("======= Toxicity Scale =======")

for n, country in enumerate(sorted(tally, key=lambda x: tally[x]), start=1):
    print(f"{n}. {country} - {tally[country]}")
```

    ======= Toxicity Scale =======
    1. EUROPE - -12318
    2. SINGAPORE - -9713
    3. US EAST - -8739
    4. US WEST - -4387
    5. AUSTRALIA - -4324
    6. AUSTRIA - -1376
    7. STOCKHOLM - -239
    8. JAPAN - -174
    9. DUBAI - -98
    10. PW TELECOM SHANGHAI - -10
    11. CHILE - -1
    12. SOUTHAFRICA - 0
    13. PW TELECOM GUANGDONG - 0
    14. PW TELECOM WUHAN - 0
    15. PW UNICOM - 0
    16. PW UNICOM TIANJIN - 0
    17. PERU - 0
    18. INDIA - 0
    19. PW TELECOM ZHEJIANG - 2
    20. BRAZIL - 69
    


```python

```
