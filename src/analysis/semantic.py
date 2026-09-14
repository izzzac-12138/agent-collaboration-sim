"""Semantic analysis of agent communication texts."""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

try:
    from gensim import corpora, models
except ImportError:
    corpora = None
    models = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None


class SemanticAnalyzer:
    """Analyze the semantic content of agent communications."""

    def __init__(self, events_df: pd.DataFrame) -> None:
        self.events_df = events_df
        self.messages_df: pd.DataFrame = pd.DataFrame()
        self._extract_messages()

    def _extract_messages(self) -> None:
        msgs = self.events_df[self.events_df["event_type"] == "message_sent"].copy()
        if msgs.empty:
            self.messages_df = msgs
            return

        records = []
        for _, row in msgs.iterrows():
            payload = row["payload"] if isinstance(row["payload"], dict) else {}
            records.append({
                "timestamp": row["timestamp"],
                "from_agent": payload.get("from_agent", row.get("agent_id", "")),
                "to_agent": payload.get("to_agent", ""),
                "content": payload.get("content", ""),
                "msg_type": payload.get("msg_type", ""),
            })
        self.messages_df = pd.DataFrame(records)

    def extract_topics_lda(self, n_topics: int = 5, passes: int = 10) -> dict:
        """Extract topics using LDA or fallback to word frequency."""
        if self.messages_df.empty:
            return {"topics": [], "topic_distribution": [], "dominant_topics_per_agent": {}}

        contents = self.messages_df["content"].tolist()

        if corpora is not None:
            tokenized = [c.lower().split() for c in contents if c]
            dictionary = corpora.Dictionary(tokenized)
            corpus = [dictionary.doc2bow(doc) for doc in tokenized]

            if len(corpus) < n_topics:
                n_topics = max(1, len(corpus))

            lda = models.LdaModel(
                corpus, num_topics=n_topics, passes=passes,
                id2word=dictionary, random_state=42,
            )

            topics = []
            for idx in range(n_topics):
                words = lda.show_topic(idx, topn=10)
                topics.append((idx, [(w, float(s)) for w, s in words]))

            topic_dist = [lda.get_document_topics(bow) for bow in corpus]

            dominant = {}
            for i, row in self.messages_df.iterrows():
                agent = row["from_agent"]
                if i < len(topic_dist) and topic_dist[i]:
                    top_topic = max(topic_dist[i], key=lambda x: x[1])[0]
                    dominant.setdefault(agent, []).append(top_topic)

            dominant_per_agent = {
                a: max(set(topics_list), key=topics_list.count)
                for a, topics_list in dominant.items()
            }

            return {
                "topics": topics,
                "topic_distribution": topic_dist,
                "dominant_topics_per_agent": dominant_per_agent,
            }

        # Fallback: word frequency
        word_counts: Counter = Counter()
        agent_words: dict[str, Counter] = {}
        for _, row in self.messages_df.iterrows():
            words = row["content"].lower().split()
            word_counts.update(words)
            agent_words.setdefault(row["from_agent"], Counter()).update(words)

        return {
            "topics": word_counts.most_common(20),
            "topic_distribution": [],
            "dominant_topics_per_agent": {
                a: wc.most_common(1)[0][0] if wc else ""
                for a, wc in agent_words.items()
            },
        }

    def compute_message_embeddings(self) -> np.ndarray:
        """Compute embeddings for message contents."""
        if self.messages_df.empty:
            return np.array([])

        contents = self.messages_df["content"].tolist()

        if SentenceTransformer is not None:
            model = SentenceTransformer("all-MiniLM-L6-v2")
            return model.encode(contents)

        # Fallback: bag-of-words embedding
        vocab: dict[str, int] = {}
        for c in contents:
            for w in c.lower().split():
                if w not in vocab:
                    vocab[w] = len(vocab)

        embeddings = np.zeros((len(contents), len(vocab) or 1))
        for i, c in enumerate(contents):
            for w in c.lower().split():
                if w in vocab:
                    embeddings[i, vocab[w]] += 1
        return embeddings

    def cluster_messages_by_topic(self, n_clusters: int = 5) -> pd.DataFrame:
        """Cluster messages by semantic similarity."""
        from sklearn.cluster import KMeans

        embeddings = self.compute_message_embeddings()
        if embeddings.size == 0:
            self.messages_df["topic_cluster"] = -1
            return self.messages_df

        n_clusters = min(n_clusters, len(embeddings))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.messages_df = self.messages_df.copy()
        self.messages_df["topic_cluster"] = kmeans.fit_predict(embeddings)
        return self.messages_df

    def get_discussion_summary(self) -> dict:
        """Overall communication pattern summary."""
        if self.messages_df.empty:
            return {"total_messages": 0}

        return {
            "total_messages": len(self.messages_df),
            "unique_senders": self.messages_df["from_agent"].nunique(),
            "unique_receivers": self.messages_df["to_agent"].nunique(),
            "avg_content_length": float(self.messages_df["content"].str.len().mean()),
            "msg_type_distribution": self.messages_df["msg_type"].value_counts().to_dict(),
            "most_active_sender": self.messages_df["from_agent"].value_counts().index[0]
            if len(self.messages_df) > 0
            else None,
        }
