import pandas as pd
from nltk.corpus import wordnet as wn
import nltk

nltk.download('wordnet')
# df = pd.read_parquet('../oct31/fashion_mentions.parquet')
df_2 =pd.read_csv('titlemeta.tsv', sep='\t', dtype=str)
hathi_metadata = df_2
df = pd.read_parquet('final_merged.parquet')
df_clean = df.drop_duplicates(subset=["term", "sentence"])
df = df_clean

# def count_colors(list, black_shades, color_adjs):
#     black_count = 0
#     color_count = 0
#     for item in list:
#         if item in black_shades:
#             black_count+=1
#         if item in color_adjs:
#             color_count+=1
#     return black_count, color_count


def count_colors(adjs, black_shades, color_adjs):
    black_count = 0
    color_count = 0

    # Handle NaN / None
    if not isinstance(adjs, list):
        return pd.Series([0, 0])

    for word in adjs:
        if not isinstance(word, str):
            continue

        w = word.strip().lower()

        if w in black_shades:
            black_count += 1
        elif w in color_adjs:
            color_count += 1

    return pd.Series([black_count, color_count])


black_shades = {'black', 'ebony', 'jet', 'onyx', 'charcoal'}


color_adjs = ['blue','orange','purple','yellow','green','red','pink','brown','navy','magenta','cyan','maroon','emerald','sage','lime','olive','gold','silver']

# def get_adj_synonyms(words):
#     synonyms = []
#     for word in words:
#         for synset in wn.synsets(word, pos=wn.ADJ):
#             if 'color' in synset.lemma():  # only color adjectives
#                 synonyms.append(lemma.name())
#         return synonyms
#
#
# from nltk.corpus import wordnet as wn

def get_adj_synonyms(words):
    synonyms = []

    for word in words:
        for synset in wn.synsets(word, pos=wn.ADJ):
            # Filter to color-related synsets
            if "color" in synset.definition().lower():
                for lemma in synset.lemmas():
                    name = lemma.name().replace("_", " ")
                    if name.lower() != word.lower():
                        synonyms.append(name)

    return synonyms


synonyms = get_adj_synonyms(color_adjs)

color_adjs.append(synonyms)



# for syn in wn.all_synsets(pos=wn.ADJ):
#     if 'color' in syn.lexname():  # only color adjectives
#         for lemma in syn.lemmas():
#             # lemma.name() is a string, not a list
#             color_adjs.append(lemma.name().replace('_', ' '))  # safe to add to set


# count_colors(
#         ['black', 'navy', 'purple', 'orange'],
#         black_shades,
#         color_adjs)

df = df.reset_index()

# df['black count', 'color_count'] = df['adjectives_fashion'].apply(count_colors, black_shades=black_shades, color_adjs=color_adjs)
df[['black count', 'color_count']] = (
    df['adjectives_fashion']
      .apply(count_colors, black_shades=black_shades, color_adjs=color_adjs)
)


# for idx, row in df.iterrows():
#     black_count, color_count = count_colors(
#         row['adjectives_fashion'],
#         black_shades,
#         color_adjs
#     )
#     df.loc[idx, 'black_count'] = black_count
#     df.loc[idx, 'color_count'] = color_count


df.to_csv('df_with_color_counts.csv')