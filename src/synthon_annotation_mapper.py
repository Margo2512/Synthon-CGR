def map_hbat_to_candidates(hbat_info, candidates, motifs_a, motifs_b):
    observed_mask = torch.zeros(len(candidates))
    
    hbat_types = set()
    for donor in hbat_info.get('donors', []):
        for motif in motifs_a:
            if donor in motif['atom_ids']:
                hbat_types.add(motif['type'])
    
    for acceptor in hbat_info.get('acceptors', []):
        for motif in motifs_b:
            if acceptor in motif['atom_ids']:
                hbat_types.add(motif['type'])
    
    for i, candidate in enumerate(candidates):
        if (candidate['left_type'] in hbat_types and 
            candidate['right_type'] in hbat_types):
            observed_mask[i] = 1.0
    
    return observed_mask