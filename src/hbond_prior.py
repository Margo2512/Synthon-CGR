import torch
import torch.nn as nn


class HBondPrior:
    def __init__(self):
        self.donor_scores = {
            'O=[C]([R])[N]([R])[R]': 1.0,   
            'O=[C](O)[R]': 0.95,            
            '[OH][Car]': 0.90,               
            '[NH2][Car]': 0.85,               
            'O=[S](=O)([R])[N]([R])[R]': 0.85,
            
            '[Nar]': 0.70,                    
            '[OH][Cal]': 0.65,                
            'C=N[N]([R])[C](=O)[R]': 0.60,   
            
            '[R][O][R]': 0.30,                
            'O=[C]([R])[O][R]': 0.25,        
            'C-H': 0.20,                     
            
            'I[R]': 0.80,                    
            'Br[R]': 0.60,                   
            'Cl[R]': 0.30,                   
            'F[R]': 0.10,                    
        }
        
        self.acceptor_scores = {
            'O=[C]([R])[N]([R])[R]': 1.0,     
            'O=[C](O)[R]': 0.95,            
            '[O]=[Car]': 0.90,              
            'O=[N+]([O-])[R]': 0.85,         
            'O=[S](=O)([R])[N]([R])[R]': 0.85, 
            
            '[Nar]': 0.75,                  
            '[R][O][R]': 0.65,               
            '[OH][Car]': 0.60,              
            'O=[C]([O-])[R]': 0.70,         
            
            'O=[C]([R])[O][R]': 0.30,         
            '[OH][Cal]': 0.25,                
            'C=N[N]([R])[C](=O)[R]': 0.20,    
            '[F][R]': 0.10,                  
        }
        
        self.compatibility = {
            ('O=[C]([R])[N]([R])[R]', 'O=[C](O)[R]'): 1.0,   
            ('O=[C](O)[R]', 'O=[C]([R])[N]([R])[R]'): 1.0,   
            ('[Nar]', '[OH][Car]'): 0.95,                     
            ('[OH][Car]', '[Nar]'): 0.95,                    
            
            ('O=[C]([R])[N]([R])[R]', '[OH][Car]'): 0.85,   
            ('[OH][Car]', 'O=[C]([R])[N]([R])[R]'): 0.85,  
            ('O=[C]([R])[N]([R])[R]', '[Nar]'): 0.80,        
            ('[Nar]', 'O=[C]([R])[N]([R])[R]'): 0.80,      
            ('O=[C](O)[R]', '[OH][Car]'): 0.75,             
            ('[OH][Car]', 'O=[C](O)[R]'): 0.75,            
            
            ('O=[C]([R])[N]([R])[R]', 'O=[S](=O)([R])[N]([R])[R]'): 0.65,
            ('O=[S](=O)([R])[N]([R])[R]', 'O=[C]([R])[N]([R])[R]'): 0.65,
            ('[Nar]', 'O=[S](=O)([R])[N]([R])[R]'): 0.60,
            ('O=[S](=O)([R])[N]([R])[R]', '[Nar]'): 0.60,
            ('[R][O][R]', '[Nar]'): 0.55,                  
            ('[Nar]', '[R][O][R]'): 0.55,
            
            ('O=[C]([R])[N]([R])[R]', '[R][O][R]'): 0.45,    
            ('[R][O][R]', 'O=[C]([R])[N]([R])[R]'): 0.45,
            ('O=[C]([R])[N]([R])[R]', 'O=[C]([O-])[R]'): 0.50, 
            ('O=[C]([O-])[R]', 'O=[C]([R])[N]([R])[R]'): 0.50,
            
            ('I[R]', '[Nar]'): 0.80,        
            ('[Nar]', 'I[R]'): 0.80,
            ('Br[R]', '[Nar]'): 0.60,        
            ('[Nar]', 'Br[R]'): 0.60,
            ('Cl[R]', '[Nar]'): 0.30,       
            ('[Nar]', 'Cl[R]'): 0.30,
            ('F[R]', '[Nar]'): 0.20,        
            ('[Nar]', 'F[R]'): 0.20,
            
            ('[Car]', '[Car]'): 0.30,        
            
            ('C-H', 'O=[C](O)[R]'): 0.20,
            ('C-H', '[R][O][R]'): 0.15,
            ('C-H', '[Nar]'): 0.10,


            ('[NH2][Car]', 'O=[C](O)[R]'): 0.90,
            ('O=[C](O)[R]', '[NH2][Car]'): 0.90,
            ('[NH2][Car]', 'O=[N+]([O-])[R]'): 0.70,
            ('O=[N+]([O-])[R]', '[NH2][Car]'): 0.70,
            ('[OH][Car]', 'O=[N+]([O-])[R]'): 0.65,
            ('O=[N+]([O-])[R]', '[OH][Car]'): 0.65,
            ('O=[C]([R])[N]([R])[R]', 'O=[C]([O-])[R]'): 0.85,
            ('O=[C]([O-])[R]', 'O=[C]([R])[N]([R])[R]'): 0.85,
            ('O=[S](=O)([R])[N]([R])[R]', 'O=[C](O)[R]'): 0.70,
            ('O=[C](O)[R]', 'O=[S](=O)([R])[N]([R])[R]'): 0.70,
            ('C=N[N]([R])[C](=O)[R]', 'O=[C](O)[R]'): 0.60,
            ('O=[C](O)[R]', 'C=N[N]([R])[C](=O)[R]'): 0.60,
            ('I[R]', '[R][O][R]'): 0.50,
            ('[R][O][R]', 'I[R]'): 0.50,
            ('Br[R]', '[R][O][R]'): 0.35,
            ('[R][O][R]', 'Br[R]'): 0.35,
        }

        import pickle, os
        _priors_path = 'Synthon-CGR/efg_hbat_priors.pkl'
        if os.path.exists(_priors_path):
            with open(_priors_path, 'rb') as f:
                _p = pickle.load(f)
            self.donor_scores.update(_p['donor_scores'])
            self.acceptor_scores.update(_p['acceptor_scores'])
            self.compatibility.update(_p['compatibility'])
            print(f"[HBondPrior] Loaded EFG-HBAT priors: "
                f"donors={len(_p['donor_scores'])}, "
                f"acceptors={len(_p['acceptor_scores'])}, "
                f"pairs={len(_p['compatibility'])//2}")
        else:
            print(f"[HBondPrior] WARNING: {_priors_path} not found")

    def get_donor_score(self, motif_type):
        return self.donor_scores.get(motif_type, 0.1)
    
    def get_acceptor_score(self, motif_type):
        return self.acceptor_scores.get(motif_type, 0.1)

    def get_compatibility(self, type_a, type_b):
        if (type_a, type_b) in self.compatibility:
            return self.compatibility[(type_a, type_b)]
        if (type_b, type_a) in self.compatibility:
            return self.compatibility[(type_b, type_a)]
        
        donor_score = self.get_donor_score(type_a)
        acceptor_score = self.get_acceptor_score(type_b)
        
        if donor_score > 0.3 and acceptor_score > 0.3:
            return 0.7 * (donor_score + acceptor_score) / 2
        
        if donor_score > 0.1 and acceptor_score > 0.1:
            return 0.3 * (donor_score + acceptor_score) / 2
        
        donor_score = self.get_donor_score(type_b)
        acceptor_score = self.get_acceptor_score(type_a)
        
        if donor_score > 0.3 and acceptor_score > 0.3:
            return 0.7 * (donor_score + acceptor_score) / 2
        
        if donor_score > 0.1 and acceptor_score > 0.1:
            return 0.3 * (donor_score + acceptor_score) / 2
        
        return 0.05