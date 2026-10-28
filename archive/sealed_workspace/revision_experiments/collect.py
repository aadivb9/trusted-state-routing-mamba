"""Periodically export a background Colab result while the host stays connected."""
import json,pathlib,subprocess,sys,time
root=pathlib.Path(__file__).resolve().parent
cell=sys.argv[1];dest=pathlib.Path(sys.argv[2])
for attempt in range(160):
 r=subprocess.run([sys.executable,str(root/'colab_control.py'),'export',cell,str(dest)],capture_output=True,text=True,timeout=125)
 print(r.stdout[-1200:],flush=True)
 if r.returncode:
  print(r.stderr[-1200:],flush=True);raise RuntimeError('Export failed; preserve local evidence')
 if (dest/'results.json').exists():break
 time.sleep(45)
else:raise TimeoutError('Export window ended')
