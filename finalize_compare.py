"""Run apply_finalization from two source trees on the real merged logits and compare encodings."""
import sys, importlib.util, numpy as np, zarr
W="/Users/george/Projects/fable-5.1-vesuvius"
def load(tag, path):
    spec=importlib.util.spec_from_file_location(f"fin_{tag}", path); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
main_mod=load("main", f"{W}/villa-main/vesuvius/src/vesuvius/models/run/finalize_outputs.py")
branch_mod=load("branch", f"{W}/villa/vesuvius/src/vesuvius/models/run/finalize_outputs.py")
merged=zarr.open(f"{W}/work/pred_rv/merged.zarr", mode="r")
coords=zarr.open(f"{W}/work/pred_rv/coordinates_part_0.zarr", mode="r")[:]
print("merged", merged.shape, merged.chunks, merged.dtype, "| patches", coords.shape[0])
C=merged.shape[0]; cz=cy=cx=int(sys.argv[1]) if len(sys.argv)>1 else merged.chunks[1]
z0,y0,x0=coords.min(axis=0); z1,y1,x1=coords.max(axis=0)+128
rows=[]
for z in range(z0, z1, cz):
    for y in range(y0, y1, cy):
        for x in range(x0, x1, cx):
            chunk=np.asarray(merged[:, z:z+cz, y:y+cy, x:x+cx], dtype=np.float32)
            if not np.any(chunk): continue
            classes=np.unique(np.argmax(chunk,axis=0))
            outs={}
            for tag,m in (("main",main_mod),("branch",branch_mod)):
                cfg=m.FinalizeConfig(mode="multiclass")
                out,empty=m.apply_finalization(chunk, C, cfg)
                if empty: outs[tag]=None; continue
                arg=out[-1]; sm=out[:C]
                # byte written for each class index present, and byte for a probability >= 0.99
                exp=np.exp(chunk-chunk.max(axis=0,keepdims=True)); p=exp/exp.sum(axis=0,keepdims=True)
                sure=p.max(axis=0)>=0.99
                outs[tag]=dict(classes={int(c): int(np.unique(arg[np.argmax(chunk,axis=0)==c])[0]) for c in classes},
                               p099_byte=(int(np.min(sm.max(axis=0)[sure])), int(np.max(sm.max(axis=0)[sure]))) if sure.any() else None)
            rows.append(((z,y,x), classes.tolist(), outs))
print(f"{len(rows)} non-empty chunks")
print(f"{'chunk (z,y,x)':>22} {'classes':>12} | main: class->byte, p>=.99 byte | branch: class->byte, p>=.99 byte")
for (zyx, cls, outs) in rows[:14]:
    m=outs['main']; b=outs['branch']
    print(f"{str(zyx):>22} {str(cls):>12} | {m['classes'] if m else None} {m['p099_byte'] if m else ''} | {b['classes'] if b else None} {b['p099_byte'] if b else ''}")
# summary: does the byte for class k vary across chunks?
for tag in ("main","branch"):
    per_class={}
    for _,_,outs in rows:
        o=outs[tag]
        if o:
            for k,v in o['classes'].items(): per_class.setdefault(k,set()).add(v)
    print(tag, "bytes used per class index across chunks:", {k: sorted(v) for k,v in sorted(per_class.items())})
