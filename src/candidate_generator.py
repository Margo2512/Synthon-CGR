from .hbond_prior import HBondPrior

_hbond_prior = HBondPrior()

def get_prior(type_a, type_b):
    return _hbond_prior.get_compatibility(type_a, type_b)

def generate_candidates(motifs_a, motifs_b, rule_library=None, max_candidates=500):
    candidates = []

    if not motifs_a or not motifs_b:
        return candidates
    
    all_pairs = []
    for i, motif_a in enumerate(motifs_a):
        for j, motif_b in enumerate(motifs_b):
            if not motif_a.get('type') or not motif_b.get('type'):
                continue
            if not motif_a.get('atom_ids') or not motif_b.get('atom_ids'):
                continue
            prior = get_prior(motif_a['type'], motif_b['type'])


            is_halogen = any(x in motif_a['type'] for x in ['F[R]', 'Cl[R]', 'Br[R]', 'I[R]']) or \
             any(x in motif_b['type'] for x in ['F[R]', 'Cl[R]', 'Br[R]', 'I[R]'])

            all_pairs.append({
                'candidate_id': len(all_pairs),
                'left_motif_id': i,
                'right_motif_id': j,
                'left_type': motif_a['type'],
                'right_type': motif_b['type'],
                'prior': prior,
                'is_halogen': is_halogen,
                'left_motif': motif_a,
                'right_motif': motif_b,
            })
    
    all_pairs.sort(key=lambda x: x['prior'], reverse=True)
    candidates = all_pairs[:max_candidates * 10] 

    print(f"DEBUG: candidates after filter: {len(candidates)}")
    
    return candidates