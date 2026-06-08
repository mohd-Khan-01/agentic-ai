import json 
import os
import numpy as np
from sentence_transformers import SentenceTransformer

"""this model converts the snetences into vectors 
similar sentences similar arrows high cosine similarity"""
print("[Memory] loading embedding model")
embedder=SentenceTransformer("all-MiniLM-L6-v2")
print("[Memory ] MOdel ready")

MEMORY_FILE="memory_store.json"

class MemoryStore:
    """this agents notebook with smart search there are two operation
    add()->writesomething in the notebook
    select()->find the most relevant pages for a query
    this is the selective primitive
    """
    def __init__(self):
        self.memories=[]
        self._load()
    
    def add(self,text:str,source:str="agent",turn:int=0):
        """sotre a new memory with its vector
    steps:
    1.first of all convert the text into vector using the embedding model
    2.next we will save all those  text,vector and the metadeta to the list
    3.persist to disk so memory survives between sessions
    """
        if not text.strip():
            return
        #convert sentence to vector (arrow in space)
        embedding=embedder.encode(
            text,convert_to_numpy=True
        ).tolist()
        
        entry={
            "text":text[500],
            "embedding":embedding,
            "source":source,
            "turn":turn
        }
        self.memories.append(entry)
        self._save()
        print(f"[Memory] Stored: '{text[:60]}...'"
              f"(total:{len(self.memories)})")
        
        # now will we wirte a function which will select the text from the notebook
        def select(self,query:str,top_k:int=3)->list:
            '''Select primitive - find top_k most relevant memories
            steps:
            1.Convert query->vector
            2.compard it with every stored vector
            3.score using cosine similarity
            4.return top_k highest scoring memories
            '''
            if not self.memories:
                return
            query_vec=embedder.encode(
                query,convert_to_numpy=True
            )
            
            scores=[]
            for mem in self.memories:
                mem_vec=np.array(mem["embedding"])
                
                #Cosine similarity -measure angle between vectors
                #dot product of normalized vectors=cos(angle)
                similarity=np.dot(query_vec,mem_vec)/(
                    np.linalg.norm(query_vec)*
                    np.linalg.norm(mem_vec)+1e-9
                )
                scores.append((
                similarity,
                mem["text"],
                mem["source"]
            ))
            scores.sort(key=lambda x: x[0], reverse=True)

            print(f"\n[SELECT] Query: '{query[:50]}'")
            for score, text, source in scores[:top_k]:
                print(f"  score={score:.3f} | "
                    f"[{source}] {text[:60]}...")

            return [text for _, text, _ in scores[:top_k]]

    def select_formatted(self, query: str, top_k: int = 3) -> str:
        """
        Same as select() but formatted for context injection.
        This is what gets injected above the user's question.
        """
        results = self.select(query, top_k)
        if not results:
            return ""

        lines = ["[RELEVANT MEMORY FROM PAST TURNS]"]
        for i, r in enumerate(results, 1):
            lines.append(f"{i}. {r}")
        lines.append("[END MEMORY]")
        return "\n".join(lines)
    def clear(self):
        self.memories = []
        self._save()
        print("[Memory] Cleared.")

    def stats(self) -> str:
        return f"[Memory] {len(self.memories)} memories stored"

    def _save(self):
        """Save memories to disk — survives between sessions."""
        with open(MEMORY_FILE, "w") as f:
            json.dump(self.memories, f, indent=2)

    def _load(self):
        """Load memories from disk on startup."""
        if os.path.exists(MEMORY_FILE):
            with open(MEMORY_FILE, "r") as f:
                self.memories = json.load(f)
            print(f"[Memory] Loaded "
                  f"{len(self.memories)} memories from disk.")
        else:
            self.memories = []


# One shared memory store for the whole session
memory = MemoryStore()    