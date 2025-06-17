import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from typing import List, Dict, Tuple, Optional, Union
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)

def evaluate_model(
    model,
    test_data: List[Dict],
    batch_size: int = 32,
    metrics: List[str] = ["similarity", "retrieval", "classification"]
) -> Dict[str, float]:
    """
    Comprehensive evaluation of embedding model
    
    Args:
        model: Trained embedding model
        test_data: Test dataset
        batch_size: Batch size for inference
        metrics: List of metrics to compute
    
    Returns:
        Dictionary of evaluation metrics
    """
    
    results = {}
    
    if "similarity" in metrics:
        sim_results = evaluate_similarity(model, test_data, batch_size)
        results.update(sim_results)
    
    if "retrieval" in metrics:
        ret_results = evaluate_retrieval(model, test_data, batch_size)
        results.update(ret_results)
    
    if "classification" in metrics:
        cls_results = evaluate_classification(model, test_data, batch_size)
        results.update(cls_results)
    
    return results

def evaluate_similarity(
    model,
    test_data: List[Dict],
    batch_size: int = 32,
    similarity_threshold: float = 0.5
) -> Dict[str, float]:
    """
    Evaluate model on similarity tasks
    """
    
    # Extract texts and labels
    texts1, texts2, true_labels = [], [], []
    
    for item in test_data:
        if "text1" in item and "text2" in item:
            texts1.append(item["text1"])
            texts2.append(item["text2"])
            true_labels.append(item.get("label", 1))
    
    if not texts1:
        logger.warning("No similarity pairs found in test data")
        return {}
    
    # Compute embeddings
    embeddings1 = model.encode(texts1, batch_size=batch_size)
    embeddings2 = model.encode(texts2, batch_size=batch_size)
    
    # Convert to numpy if needed
    if torch.is_tensor(embeddings1):
        embeddings1 = embeddings1.cpu().numpy()
    if torch.is_tensor(embeddings2):
        embeddings2 = embeddings2.cpu().numpy()
    
    # Compute cosine similarities
    similarities = np.array([
        cosine_similarity([emb1], [emb2])[0][0] 
        for emb1, emb2 in zip(embeddings1, embeddings2)
    ])
    
    # Convert similarities to binary predictions
    predictions = (similarities > similarity_threshold).astype(int)
    
    # Compute metrics
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, average='binary')
    recall = recall_score(true_labels, predictions, average='binary')
    f1 = f1_score(true_labels, predictions, average='binary')
    
    # Correlation with true labels
    correlation = np.corrcoef(similarities, true_labels)[0, 1]
    
    return {
        "similarity_accuracy": accuracy,
        "similarity_precision": precision,
        "similarity_recall": recall,
        "similarity_f1": f1,
        "similarity_correlation": correlation,
        "mean_similarity": np.mean(similarities),
        "std_similarity": np.std(similarities)
    }

def evaluate_retrieval(
    model,
    test_data: List[Dict],
    batch_size: int = 32,
    k_values: List[int] = [1, 5, 10]
) -> Dict[str, float]:
    """
    Evaluate model on retrieval tasks
    """
    
    # Prepare queries and corpus
    queries = []
    corpus = []
    relevance_map = {}
    
    for i, item in enumerate(test_data):
        if "query" in item and "documents" in item:
            queries.append(item["query"])
            docs = item["documents"]
            
            # Add documents to corpus
            start_idx = len(corpus)
            corpus.extend(docs)
            
            # Create relevance mapping
            relevance_map[i] = {
                "start_idx": start_idx,
                "relevant_docs": item.get("relevant_indices", [0])  # Assume first doc is relevant
            }
    
    if not queries:
        logger.warning("No retrieval queries found in test data")
        return {}
    
    # Encode queries and corpus
    logger.info("Encoding queries...")
    query_embeddings = model.encode(queries, batch_size=batch_size)
    
    logger.info("Encoding corpus...")
    corpus_embeddings = model.encode(corpus, batch_size=batch_size)
    
    # Convert to numpy
    if torch.is_tensor(query_embeddings):
        query_embeddings = query_embeddings.cpu().numpy()
    if torch.is_tensor(corpus_embeddings):
        corpus_embeddings = corpus_embeddings.cpu().numpy()
    
    # Compute retrieval metrics
    results = {}
    
    for k in k_values:
        hits_at_k = 0
        mrr_scores = []
        
        for i, query_emb in enumerate(query_embeddings):
            if i not in relevance_map:
                continue
            
            # Get document range for this query
            start_idx = relevance_map[i]["start_idx"]
            num_docs = len(test_data[i]["documents"])
            end_idx = start_idx + num_docs
            
            # Compute similarities with documents for this query
            doc_embeddings = corpus_embeddings[start_idx:end_idx]
            similarities = cosine_similarity([query_emb], doc_embeddings)[0]
            
            # Get top-k documents
            top_k_indices = np.argsort(similarities)[::-1][:k]
            
            # Check if any relevant document is in top-k
            relevant_docs = relevance_map[i]["relevant_docs"]
            hits = any(idx in relevant_docs for idx in top_k_indices)
            hits_at_k += hits
            
            # Compute MRR
            for rank, idx in enumerate(top_k_indices):
                if idx in relevant_docs:
                    mrr_scores.append(1.0 / (rank + 1))
                    break
            else:
                mrr_scores.append(0.0)
        
        # Calculate metrics
        hits_at_k_score = hits_at_k / len(queries) if queries else 0
        mrr_score = np.mean(mrr_scores) if mrr_scores else 0
        
        results[f"hits_at_{k}"] = hits_at_k_score
        results[f"mrr_at_{k}"] = mrr_score
    
    return results

def evaluate_classification(
    model,
    test_data: List[Dict],
    batch_size: int = 32,
    use_knn: bool = True,
    k_neighbors: int = 5
) -> Dict[str, float]:
    """
    Evaluate model on classification tasks using embeddings
    """
    
    # Extract texts and labels
    texts = []
    labels = []
    
    for item in test_data:
        if "text" in item and "label" in item:
            texts.append(item["text"])
            labels.append(item["label"])
    
    if not texts:
        logger.warning("No classification data found in test data")
        return {}
    
    # Encode texts
    embeddings = model.encode(texts, batch_size=batch_size)
    
    if torch.is_tensor(embeddings):
        embeddings = embeddings.cpu().numpy()
    
    # Simple classification using k-NN or centroid-based approach
    if use_knn:
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.model_selection import cross_val_score
        
        # Use cross-validation for evaluation
        knn = KNeighborsClassifier(n_neighbors=k_neighbors)
        cv_scores = cross_val_score(knn, embeddings, labels, cv=5, scoring='accuracy')
        
        return {
            "classification_accuracy": np.mean(cv_scores),
            "classification_std": np.std(cv_scores)
        }
    else:
        # Centroid-based classification
        from sklearn.cluster import KMeans
        from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
        
        # Determine number of clusters
        unique_labels = len(set(labels))
        
        # Perform clustering
        kmeans = KMeans(n_clusters=unique_labels, random_state=42)
        cluster_labels = kmeans.fit_predict(embeddings)
        
        # Compute clustering metrics
        ari = adjusted_rand_score(labels, cluster_labels)
        nmi = normalized_mutual_info_score(labels, cluster_labels)
        
        return {
            "classification_ari": ari,
            "classification_nmi": nmi
        }

def compute_embedding_quality_metrics(
    embeddings: np.ndarray,
    labels: Optional[List] = None
) -> Dict[str, float]:
    """
    Compute intrinsic quality metrics for embeddings
    """
    
    metrics = {}
    
    # Embedding statistics
    metrics["embedding_mean_norm"] = np.mean(np.linalg.norm(embeddings, axis=1))
    metrics["embedding_std_norm"] = np.std(np.linalg.norm(embeddings, axis=1))
    
    # Cosine similarity statistics
    similarities = cosine_similarity(embeddings)
    np.fill_diagonal(similarities, np.nan)  # Exclude self-similarities
    
    metrics["mean_cosine_similarity"] = np.nanmean(similarities)
    metrics["std_cosine_similarity"] = np.nanstd(similarities)
    
    # Embedding diversity (average pairwise distance)
    distances = 1 - similarities
    metrics["mean_pairwise_distance"] = np.nanmean(distances)
    
    # If labels are provided, compute intra/inter-class similarities
    if labels is not None:
        unique_labels = list(set(labels))
        intra_class_sims = []
        inter_class_sims = []
        
        for label in unique_labels:
            label_indices = [i for i, l in enumerate(labels) if l == label]
            
            if len(label_indices) > 1:
                # Intra-class similarities
                for i in range(len(label_indices)):
                    for j in range(i + 1, len(label_indices)):
                        idx1, idx2 = label_indices[i], label_indices[j]
                        intra_class_sims.append(similarities[idx1, idx2])
            
            # Inter-class similarities
            other_indices = [i for i, l in enumerate(labels) if l != label]
            for idx1 in label_indices:
                for idx2 in other_indices:
                    inter_class_sims.append(similarities[idx1, idx2])
        
        if intra_class_sims:
            metrics["mean_intra_class_similarity"] = np.mean(intra_class_sims)
        if inter_class_sims:
            metrics["mean_inter_class_similarity"] = np.mean(inter_class_sims)
        
        # Silhouette-like score
        if intra_class_sims and inter_class_sims:
            metrics["class_separation_score"] = (
                np.mean(intra_class_sims) - np.mean(inter_class_sims)
            )
    
    return metrics