# pdbs/

Place your truncated target PDB files here. Files in this directory are gitignored (except this README and `.gitkeep`) so your target structures do not get committed.

Naming convention used in `nanocdr3gen.py`:

```
pdbs/<target_name>_trunc.pdb
```

The PDB should contain a single chain of the target region around your hotspots, truncated to a 25 Å sphere. See the `examples/` folder and the truncation snippet in the main README for how to generate this file.
