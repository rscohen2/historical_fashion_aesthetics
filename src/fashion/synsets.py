import itertools
from nltk.corpus import wordnet
from nltk.corpus.reader.wordnet import Synset


def get_leaves(synset) -> list[Synset]:
    """
    Get the leaves of a synset in the WordNet hierarchy.

    Args:
        synset: A WordNet synset object.

    Returns:
        A list of leaves (synsets) under the given synset.
    """
    if not isinstance(synset, Synset):
        raise ValueError("Input must be a WordNet synset.")
    if not synset.hyponyms():
        return [synset]
    leaves = []
    for hyponym in synset.hyponyms():
        leaves.extend(get_leaves(hyponym))
    return leaves


if __name__ == "__main__":
    clothing = wordnet.synset("clothing.n.01")

    if not isinstance(clothing, Synset):
        raise ValueError("The specified synset is not valid.")

    # Get all leaves under the clothing synset
    articles = clothing.closure(lambda x: x.hyponyms())
    article_terms = list(
        itertools.chain.from_iterable(
            [
                [lemma.replace("_", " ") for lemma in synset.lemma_names()]
                for synset in articles
            ]
        )
    )

    print("\n".join(article_terms))
