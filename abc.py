import mdtraj as md

t = md.load("sample_data/8H0R/traj.xtc", top="sample_data/8H0R/topology_fixed.pdb")

print(t.n_atoms)
print(t.topology)
