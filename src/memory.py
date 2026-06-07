import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer

"""
this model converts the sentence into vectors
"all-MINILM-L-V2"
"""

print("[Memory] loading embedding model...")
embedder=SentenceTransformer("all-MiniLM-L6-v2")
print("[Memory] Model ready.")
MEMORY_FILE = "memory_store.json"

class  MemoryStore:
    """
    this is the agents smart notebook
    add()->write something in the notebook
    select()-> find the most relevant pages
    
    """
    
    def __init__(self):
        self.memories=[]
        self.load()
    def add(self,text:str ,source:str="agent",turn:int=0):
        """store a new memory
        converts text-> vector and saves both
        """
        
        if not text.strip():
            return 
        embedding=embedder.encode(text,convert_to_numpy=True).tolist()
        entry={
            "text":text[:500],
            "embedding":embedding,
            "source":source,
            "turn":turn,
        }
        self.memories.append(entry)
        self._save()
        print(f"[Memory] Stored: '{text[:60]}'"
              f"(total:{len(self.memories)})")
        
        def select(self,query:str,top_k: int=3)->list:
            """
            retrieve top_k most relevant memories for a query.
            
            steps:
            1.compare query to vector 
            2.compare with every stores memory vector 
            3.return the closest one (highest cosine similarity)
                """
            if not self.memories:
                return []
            
            query_vec=embedder.encode(query,convert_to_numpy=True)
            
            scores=[]
            
            for mem in self.memories:
                mem_vec=np.array(mem["embedding"])
                #cosine similarity - measure angle between them vectors
                #close to 1.0 =similar meaning  ,close to 0 =different
                similarity=np.dot(query_vec,mem_vec)/(
                    np.linalog.norm(query_vec)*np.linalig.norm(mem_vec)+1e-9
                )
                scores.append((similarity,mem['text'],mem['source']))
            
            