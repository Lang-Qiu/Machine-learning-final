import re, glob, os
base = r"E:\LQiu\lab_folder\Machine_learning\AGID_Project\Stage2_Writing\draft"
tot = 0
for f in sorted(glob.glob(os.path.join(base, "0*.md"))):
    t = open(f, encoding="utf-8").read()
    t = re.sub(r"<!--.*?-->", "", t, flags=re.S)   # drop draft comments
    t = re.sub(r"^\|.*$", "", t, flags=re.M)         # drop table rows
    t = re.sub(r"[#*_`>]", "", t)                    # drop md markup chars
    w = len(t.split())
    tot += w
    print("%-30s %5d words" % (os.path.basename(f), w))
print("-" * 42)
print("%-30s %5d words" % ("TOTAL (prose, incl. abstract)", tot))
