def reaction_center_prediction(device, model, mol_graphs, complete_graphs):
    
    mol_graphs = mol_graphs.to(device)
    complete_graphs = complete_graphs.to(device)
    node_feats = mol_graphs.ndata.pop('hv').to(device)
    edge_feats = mol_graphs.edata.pop('he').to(device)
    node_pair_feats = complete_graphs.edata.pop('feats').to(device)

    return model(mol_graphs, complete_graphs, node_feats, edge_feats, node_pair_feats)