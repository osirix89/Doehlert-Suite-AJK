#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Ejecuta pruebas nuevas de dominio, optimización y multirrespuesta y genera JSON."""
import json, subprocess, sys, datetime
from pathlib import Path
root=Path(__file__).resolve().parents[1]
cmd=[sys.executable,'-m','unittest','discover','-s',str(root/'auditor'/'tests'),'-v']
run=subprocess.run(cmd,cwd=root,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
report={'timestamp':datetime.datetime.now().astimezone().isoformat(),'command':cmd,
        'returncode':run.returncode,'output':run.stdout,
        'status':'PASS' if run.returncode==0 else 'FAIL'}
outdir=root/'Exports'/datetime.datetime.now().astimezone().strftime('%Y-%m-%d')/'Auditoria_Tecnica'
outdir.mkdir(parents=True, exist_ok=True)
out=outdir/'extended_audit_report.json'
out.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(run.stdout); print(f"\nEstado: {report['status']} · informe: {out}")
sys.exit(run.returncode)
