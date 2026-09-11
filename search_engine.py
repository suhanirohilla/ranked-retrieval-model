import re
import math
import os
import json
from collections import defaultdict

# ==========================================
# PART A: Corpus & Pre-processing
# ==========================================

# Stop-Word Policy Justification:
# We use a custom, consistent list of highly frequent English stop words.
# Justification: Words like "a", "the", "is", and "for" carry no semantic meaning 
# regarding clothing attributes. Removing them reduces the index size and prevents 
# artificially inflated document frequencies, improving retrieval precision.
STOP_WORDS = {"a", "an", "the", "and", "or", "but", "is", "are", "was", "were", 
              "for", "with", "from", "this", "it", "to", "in", "on", "of"}

def simple_stem(word):
    """
    A lightweight custom stemmer for clothing terms to avoid third-party dependencies.
    """
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith("ing") and len(word) > 4:
        return word[:-3]
    if word.endswith("ed") and len(word) > 3:
        return word[:-2]
    if word.endswith("s") and not word.endswith("ss") and len(word) > 3:
        return word[:-1]
    return word

def tokenize_and_normalize(text):
    """
    Tokenizes text, normalizes case, removes punctuation, stop-words, and applies stemming.
    """
    # Remove punctuation and tokenize
    tokens = re.findall(r'\b\w+\b', text.lower())
    processed_tokens = []
    for token in tokens:
        if token not in STOP_WORDS:
            stemmed = simple_stem(token)
            processed_tokens.append(stemmed)
    return processed_tokens

def build_index(filename):
    """
    Reads corpus_100.txt and builds a positional inverted index.
    """
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()

    docs = re.findall(r'<DOC>\s*<DOCID>(.*?)</DOCID>\s*<CATEGORY>(.*?)</CATEGORY>\s*<TITLE>(.*?)</TITLE>\s*<TEXT>(.*?)</TEXT>\s*</DOC>', content, re.DOTALL)
    
    # positional_index structure: {term: {'df': int, 'postings': {docID: [pos1, pos2, ...]}}}
    positional_index = defaultdict(lambda: {'df': 0, 'postings': defaultdict(list)})
    doc_metadata = {}
    
    for doc_id, category, title, text in docs:
        doc_id = doc_id.strip()
        doc_metadata[doc_id] = {'title': title.strip(), 'category': category.strip()}
        
        tokens = tokenize_and_normalize(text)
        
        # Build positional index
        for pos, term in enumerate(tokens):
            positional_index[term]['postings'][doc_id].append(pos)
            
    # Calculate Document Frequency (df)
    for term in positional_index:
        positional_index[term]['df'] = len(positional_index[term]['postings'])
        
    return positional_index, doc_metadata, len(docs)

# ==========================================
# PART B: Vector Space Model (lnc.ltc)
# ==========================================

def get_doc_lengths(positional_index, N):
    """
    Computes the cosine normalization length for each document using lnc weighting.
    wd,t = 1 + log10(tf)
    """
    doc_lengths = defaultdict(float)
    for term, data in positional_index.items():
        for doc_id, positions in data['postings'].items():
            tf = len(positions)
            w_dt = 1 + math.log10(tf)
            doc_lengths[doc_id] += w_dt ** 2
            
    for doc_id in doc_lengths:
        doc_lengths[doc_id] = math.sqrt(doc_lengths[doc_id])
    return doc_lengths

def vsm_search(query, positional_index, doc_lengths, doc_metadata, N):
    """
    Ranked retrieval using lnc.ltc cosine similarity.
    """
    query_tokens = tokenize_and_normalize(query)
    if not query_tokens:
        return []

    # Calculate query tf
    query_tf = defaultdict(int)
    for token in query_tokens:
        query_tf[token] += 1

    query_weights = {}
    query_norm_sq = 0.0
    
    # ltc weighting for query
    # wq,t = (1 + log10(tf)) * log10(N/df)
    for term, tf in query_tf.items():
        if term in positional_index:
            df = positional_index[term]['df']
            idf = math.log10(N / df)
            w_qt = (1 + math.log10(tf)) * idf
            query_weights[term] = w_qt
            query_norm_sq += w_qt ** 2
            
    query_norm = math.sqrt(query_norm_sq)
    if query_norm == 0:
        return []

    scores = defaultdict(float)
    
    for term, w_qt in query_weights.items():
        for doc_id, positions in positional_index[term]['postings'].items():
            tf_doc = len(positions)
            w_dt = 1 + math.log10(tf_doc)
            scores[doc_id] += w_dt * w_qt

    # Normalize scores
    results = []
    for doc_id, score in scores.items():
        normalized_score = score / (doc_lengths[doc_id] * query_norm)
        results.append((doc_id, normalized_score, doc_metadata[doc_id]['title'], doc_metadata[doc_id]['category']))

    # Sort descending by score, ascending by docID to break ties
    results.sort(key=lambda x: (-x[1], x[0]))
    return results[:10]

# ==========================================
# PART C: Positional Index Search
# ==========================================

def positional_search(query, positional_index, doc_metadata, search_type="phrase", k=1):
    """
    Executes exact phrase search or proximity search using positions.
    """
    tokens = tokenize_and_normalize(query)
    if len(tokens) < 2:
        return []

    # Get documents containing all terms
    common_docs = set(positional_index[tokens[0]]['postings'].keys())
    for token in tokens[1:]:
        if token not in positional_index:
            return []
        common_docs.intersection_update(positional_index[token]['postings'].keys())

    results = []
    for doc_id in common_docs:
        # Get positions for the first term
        valid_positions = [[pos] for pos in positional_index[tokens[0]]['postings'][doc_id]]
        
        for i in range(1, len(tokens)):
            next_term_positions = positional_index[tokens[i]]['postings'][doc_id]
            new_valid_positions = []
            
            for path in valid_positions:
                last_pos = path[-1]
                for next_pos in next_term_positions:
                    if search_type == "phrase":
                        if next_pos == last_pos + 1:
                            new_valid_positions.append(path + [next_pos])
                    elif search_type == "proximity":
                        if 0 < (next_pos - last_pos) <= k:
                            new_valid_positions.append(path + [next_pos])
            valid_positions = new_valid_positions
            
        if valid_positions:
            results.append({
                'doc_id': doc_id,
                'title': doc_metadata[doc_id]['title'],
                'category': doc_metadata[doc_id]['category'],
                'matches': valid_positions
            })
            
    # Sort by doc_id ascending
    results.sort(key=lambda x: x['doc_id'])
    return results[:10]

# ==========================================
# PART D & E: Application & Testing
# ==========================================

def run_tests(positional_index, doc_lengths, doc_metadata, N):
    print("\n" + "="*50)
    print("PART E: AUTOMATED TESTING")
    print("="*50)
    
    print("\n--- 1. Free-Text Queries (VSM) ---")
    free_text_queries = ["cotton shirt", "black jeans", "festive saree", "winter jacket", "women dress", 
                         "oversized hoodie", "slim fit", "regular fit kurta", "stretch leggings", "blue sweatshirt"]
    for q in free_text_queries:
        res = vsm_search(q, positional_index, doc_lengths, doc_metadata, N)
        print(f"Query: '{q}' | Top Result: {res[0][0] if res else 'None'} ({res[0][1]:.4f})" if res else f"Query: '{q}' | No matches")

    print("\n--- 2. Exact Phrase Queries ---")
    phrase_queries = ["cotton shirt", "stretch denim", "festive wear", "winter wear", "regular fit"]
    for q in phrase_queries:
        res = positional_search(q, positional_index, doc_metadata, search_type="phrase")
        print(f"Phrase: '{q}' | Found in {len(res)} docs. First match: {res[0]['doc_id'] if res else 'None'}")
        if res:
            print(f"  -> Positions found in {res[0]['doc_id']}: {res[0]['matches']}")

    print("\n--- 3. Proximity Queries ---")
    prox_queries = [("cotton shirt", 3), ("stretch denim", 4), ("winter wear", 3)]
    for q, k in prox_queries:
        res = positional_search(q, positional_index, doc_metadata, search_type="proximity", k=k)
        print(f"Proximity: '{q}' WITHIN/{k} | Found in {len(res)} docs.")
        if res:
            print(f"  -> Positions found in {res[0]['doc_id']}: {res[0]['matches']}")

    print("\n--- 4. Missing Term Query ---")
    res = vsm_search("spacesuit", positional_index, doc_lengths, doc_metadata, N)
    print(f"Query: 'spacesuit' | Results: {len(res)} (Expected 0)")

def main():
    if not os.path.exists("corpus_100.txt"):
        print("Error: 'corpus_100.txt' not found in current directory.")
        return

    print("Building positional index from corpus_100.txt...")
    positional_index, doc_metadata, N = build_index("corpus_100.txt")
    doc_lengths = get_doc_lengths(positional_index, N)
    print(f"Index built successfully! {N} documents processed.\n")

    # Run Part E tests
    run_tests(positional_index, doc_lengths, doc_metadata, N)

    print("\n" + "="*50)
    print("PART D: CLOTHING SEARCH ENGINE CLI")
    print("="*50)
    
    # Save Dictionary/Inverted Index output
    with open("dictionary_output.txt", "w") as f:
        json.dump({k: v['df'] for k, v in positional_index.items()}, f, indent=4)
        
    # Save Positional Index output
    with open("positional_output.txt", "w") as f:
        json.dump({k: v['postings'] for k, v in positional_index.items()}, f, indent=4)
    print("Output files (dictionary_output.txt, positional_output.txt) generated successfully!")

    while True:
        print("\nSelect Search Mode:")
        print("1. Free-Text Search (VSM)")
        print("2. Exact Phrase Search")
        print("3. Proximity Search")
        print("4. Exit")
        choice = input("Enter choice (1-4): ")

        if choice == '4':
            break

        query = input("Enter your clothing query: ")
        
        if choice == '1':
            results = vsm_search(query, positional_index, doc_lengths, doc_metadata, N)
            if not results:
                print(f"\nNo results found for '{query}'.")
            else:
                print(f"\nTop {len(results)} Results for '{query}':")
                for rank, (doc_id, score, title, category) in enumerate(results, 1):
                    print(f"{rank}. [DocID: {doc_id}] Score: {score:.4f} | {category} | {title}")
                
        elif choice == '2':
            results = positional_search(query, positional_index, doc_metadata, search_type="phrase")
            if not results:
                print(f"\nNo phrase matches found for '{query}'.")
            else:
                print(f"\nTop {len(results)} Phrase Results for '{query}':")
                for rank, res in enumerate(results, 1):
                    print(f"{rank}. [DocID: {res['doc_id']}] {res['category']} | {res['title']} | Positions: {res['matches']}")
                
        elif choice == '3':
            k = int(input("Enter proximity distance (k): "))
            results = positional_search(query, positional_index, doc_metadata, search_type="proximity", k=k)
            if not results:
                print(f"\nNo proximity matches found for '{query}' WITHIN/{k}.")
            else:
                print(f"\nTop {len(results)} Proximity Results for '{query}' WITHIN/{k}:")
                for rank, res in enumerate(results, 1):
                    print(f"{rank}. [DocID: {res['doc_id']}] {res['category']} | {res['title']} | Positions: {res['matches']}")

if __name__ == "__main__":
    main()