import sys, numpy as np
nrm=np.linalg.norm
def mr(a,b): return nrm(a)/nrm(b), nrm(a/nrm(a)-b/nrm(b))
runs={k:dict(np.load(f)) for k,f in (a.split('=') for a in sys.argv[1:])}
old=np.load('/Users/olivercase/iNOB/outputs/_superseded_pre_sign_fix_2026-11-25/duneuro_leadfield_spine.npz',allow_pickle=True)
ref_key=list(runs)[-1]; ref=runs[ref_key]; Lref=(ref['pri__multipolar_venant']-ref['sec__multipolar_venant'])*1e6
print('reference: %s multipolar_venant (%d tets); sources x 3 moments, %d coils'%(ref_key,ref['n_tets'],len(ref['idx'])))
print('%-10s %-20s %9s | all sources vs ref: MAG  RDM | #72 z-moment peak fT/nAm | z-moment peak median over sources | sec/pri'%('mesh','model','tets'))
for k,r in runs.items():
    for m in ('partial_integration','venant','multipolar_venant'):
        if 'pri__'+m not in r: continue
        p,s=r['pri__'+m]*1e6,r['sec__'+m]*1e6; L=p-s
        mag,rdm=mr(L,Lref); z=L[:,2::3]
        print('%-10s %-20s %9d |                    %5.2f %5.2f | %6.2f                   | %6.2f                            | %.3f'%(k,m,r['n_tets'],mag,rdm,np.abs(z[:,72]).max(),np.median(np.abs(z).max(0)),nrm(s)/nrm(p)))
r=runs[list(runs)[0]]; idx=r['idx']
Lo=old['L'][idx]*1e6; p=r['pri__partial_integration']*1e6
print('stored old field on these coils equals pri+sec of base PI: MAG %.4f RDM %.4f'%mr(Lo,p+r['sec__partial_integration']*1e6))
print('free-space #72 z peak %.2f ; old stored %.2f'%(np.abs(p[:,3*72+2]).max(),np.abs(Lo[:,3*72+2]).max()))
