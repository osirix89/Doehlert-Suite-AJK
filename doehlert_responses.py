# -*- coding: utf-8 -*-
# SPDX-License-Identifier: GPL-3.0-only
"""Lectura de CSV ancho con múltiples respuestas Y1...Yn, compatible con CSV previo."""
from __future__ import annotations
import csv
from dataclasses import dataclass
import numpy as np

@dataclass
class MultiResponseData:
    headers: list
    factor_names: list
    real: np.ndarray
    coded: np.ndarray
    responses: dict
    run_names: list
    delimiter: str

def _norm(s):
    return ''.join(c.lower() for c in str(s).strip().replace(' ','_') if c.isalnum() or c=='_')

def read_multiresponse_csv(path):
    with open(path,'r',encoding='utf-8-sig',newline='') as f: lines=f.read().splitlines()
    if not lines: raise ValueError('CSV vacío.')
    delim=None
    if lines[0].strip().lower().startswith('sep='):
        delim=lines.pop(0).strip()[4:5] or ','
    if delim is None:
        try: delim=csv.Sniffer().sniff('\n'.join(lines[:20]),delimiters=',;\t').delimiter
        except csv.Error: delim=','
    rows=list(csv.reader(lines,delimiter=delim))
    if len(rows)<12: raise ValueError('Se requieren encabezado y al menos 11 filas; el ajuste cuadrático completo requiere más de 10 parámetros.')
    headers=[h.strip() for h in rows[0]]; norm=[_norm(h) for h in headers]
    run_idx=next((i for i,n in enumerate(norm) if n in ('corrida','run','run_id',' ensayo','id')),None)
    coded_idx=[next((i for i,n in enumerate(norm) if n in (f'x{k}_cod',f'x{k}_coded',f'x{k}_codificado')),None) for k in (1,2,3)]
    response_idx=[i for i,n in enumerate(norm) if n.startswith(('y','respuesta','response','resultado')) and not n.endswith(('_cod','_coded'))]
    if not response_idx:
        raise ValueError('No se detectaron respuestas. Nombra las columnas Y1, Y2... o Respuesta_Y.')
    excluded=set(response_idx)|set(i for i in coded_idx if i is not None)
    if run_idx is not None: excluded.add(run_idx)
    factor_idx=[i for i in range(len(headers)) if i not in excluded]
    if len(factor_idx)<3: raise ValueError('No se identificaron tres columnas de factores.')
    factor_idx=factor_idx[:3]
    def number(row,i,line):
        try: v=float(row[i].strip().replace(',','.'))
        except Exception as e: raise ValueError(f'Fila {line}: valor no numérico en {headers[i]!r}.') from e
        if not np.isfinite(v): raise ValueError(f'Fila {line}: valor no finito en {headers[i]!r}.')
        return v
    real=[]; coded=[]; resp={headers[i]:[] for i in response_idx}; run=[]
    for ln,row in enumerate(rows[1:],2):
        if not row or all(not c.strip() for c in row): continue
        if len(row)<len(headers): raise ValueError(f'Fila {ln}: tiene menos columnas que el encabezado.')
        real.append([number(row,i,ln) for i in factor_idx])
        if all(i is not None for i in coded_idx): coded.append([number(row,i,ln) for i in coded_idx])
        for i in response_idx: resp[headers[i]].append(number(row,i,ln))
        run.append(row[run_idx].strip() if run_idx is not None else str(len(run)+1))
    real=np.asarray(real,float)
    if len(real)<11: raise ValueError('El modelo cuadrático completo necesita al menos 11 corridas.')
    if all(i is None for i in coded_idx):
        from doehlert_engine import ALPHA
        coded=np.column_stack([(real[:,j]-(real[:,j].min()+real[:,j].max())/2)/((real[:,j].max()-real[:,j].min())/2/ALPHA[j]) for j in range(3)])
    else: coded=np.asarray(coded,float)
    if not np.all(np.isfinite(coded)): raise ValueError('Hay coordenadas codificadas no finitas.')
    return MultiResponseData(headers,[headers[i] for i in factor_idx],real,coded,
                             {k:np.asarray(v,float) for k,v in resp.items()},run,delim)
