def cache_embeddings(embedding_key, embedding_value, cache_dict):
    """Caches the embeddings in the provided cache dictionary."""
    cache_dict[embedding_key] = embedding_value

def get_cached_embedding(embedding_key, cache_dict):
    """Retrieves the cached embedding if it exists, otherwise returns None."""
    return cache_dict.get(embedding_key, None)

def clear_cache(cache_dict):
    """Clears the entire cache."""
    cache_dict.clear()

# Initialize an empty cache dictionary
embedding_cache = {}