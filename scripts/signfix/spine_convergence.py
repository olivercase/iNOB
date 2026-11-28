"""Mesh / source-model convergence of the corrected spine MEG field on a coil subset.
usage: conv.py FEM.mat OUT.npz [n_sites]"""
import sys, os, dataclasses, numpy as np
from types import SimpleNamespace
from inob.cli._common import apply_source_target
from inob.config import load_config, SOURCE_MODEL_TYPES, SourceModelCfg
from inob.io.hdf5 import load_fem, load_sensors
from inob.forward.duneuro_driver import (import_duneuro, build_driver, attach_coils, build_orthogonal_dipoles,
                                         build_source_model_config, compute_meg_field_components)
fem_path, out = sys.argv[1], sys.argv[2]; n_sites = int(sys.argv[3]) if len(sys.argv) > 3 else 60
R = '/Users/olivercase/iNOB'
cfg = apply_source_target(load_config(os.path.join(R, 'configs', 'spine.yaml')), 'spine')
fem = load_fem(fem_path); sens = load_sensors(cfg.outputs.sensors_mat)
old = np.load(os.path.join(R, 'outputs/_superseded_pre_sign_fix_2026-11-25/duneuro_leadfield_spine.npz'), allow_pickle=True)
src = old['source_pos']; assert np.allclose(old['coil_pos'], sens.coilpos)
pos = sens.coilpos; n = len(pos) // 3; site = pos[:n]
assert np.allclose(pos[:n], pos[n:2 * n]) and np.allclose(pos[:n], pos[2 * n:])
d = np.linalg.norm(site - src[72], axis=1); pick = np.sort(np.argsort(d)[:n_sites])
idx = np.concatenate([pick, pick + n, pick + 2 * n])
dp = import_duneuro(cfg); driver, dcfg, cond = build_driver(cfg, fem)
attach_coils(driver, dp, sens.coilpos[idx], sens.coilori[idx])
T = np.array(driver.computeMEGTransferMatrix(dcfg)[0])
dips = build_orthogonal_dipoles(dp, src)
save = dict(idx=idx, n_tets=len(fem.tets), src=src)
for m in SOURCE_MODEL_TYPES:
    try:
        dcfg['source_model'] = build_source_model_config(SimpleNamespace(forward=SimpleNamespace(source_model=SourceModelCfg(type=m))))
        p, s = compute_meg_field_components(driver, T, dips, dcfg)
        save['pri__' + m], save['sec__' + m] = p, s
        print('OK', m, flush=True)
    except Exception as e:
        print('FAILED', m, repr(e)[:300], flush=True)
    np.savez(out, **save)
print('CONV DONE', len(fem.tets), flush=True); os._exit(0)
