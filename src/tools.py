import math
import json
import urllib.request
import urllib.parse #used to safely encode URLs

def calculator(expression: str)-> str:
    try:
        allowed_names={
            k: v for k,v in math.__dict__.items()
            if not k.startswith("_") #it removes internal python stuff like __doc__,__loader__
            
        }
        result=eval(expression,{"__builtins__":{}},allowed_names) #__builtins__" this remove acces to dangerous python built ins without this users could do malicious things 
        return str(round(result,6))#returning output upto 6 decimals
    except Exception as e:
        return f"calculator error {e}"
    
def wikipedia_search(query: str) -> str:
    try:
        encoded=urllib.parse.quote(query.replace("","_"))#Wikipedia URLs use underscores:
        url=f"https://en.wikipedia.org/api/rest_v1/page/summary/{encoded}"
        req=urllib.request.Request(
            url,
            headers={"User-Agent":"AgenticAI-Learning-Project/1.0"}
        )
        with urllib.request.urlopen(req,timeout=6) as response:
            data=json.loads(response.read().decode())
            extract=data.get("extract","NO summary Found")
            return extract[:800]
    except urllib.error.HTTPError as e:
        return f"wikepidia page not found  ({e.code})"
    except Exception as e:
        return f"wikepidia error{e}"
    
TOOLS={
    "calculator" : {
        "fn" : calculator,
        "description":"evaluates the math expression,Input: a math string like 'sqrt(144) + 2**8'",
        
    },
    
        "wikipedia_search" :{
            "fn":wikipedia_search,
            "description":"Searches a wikipedia,Input: a search query like 'transformer neural network'"
            
        }
    }
    
def dispatch_tool(name:str,input_str:str)->str:
    """it calls the right tool by name if doesn't exist will 
    return an error"""
    
    if name not in TOOLS:
        available=", ".join(TOOLS.keys())
        return f"Error:tool{name} not found , Available:{available}"
    return TOOLS[name]["fn"](input_str)
def get_description()->str:
    """formats tool description for injection into system prompt"""
    lines=[]
    for name,meta in TOOLS.items():
        lines.append(f"-{name}:{meta['description']}")
    return "\n".join(lines)